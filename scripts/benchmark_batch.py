#!/usr/bin/env python3
"""Official-API batch-inference benchmark for qwen3-tts-rocm (v0.2.1 Task 3).

Measures what the OFFICIAL qwen-tts API does with list-of-texts batching on
the Radeon 8060S (gfx1151) — no custom batching layer anywhere. Each family
uses its ONLY supported official entry point, exactly as documented:

* ``custom-voice`` / ``custom-voice-0.6b``:
  ``generate_custom_voice(text=[...], language="Auto", speaker=spk)``
* ``voice-design``:
  ``generate_voice_design(text=[...], instruct=..., language="Auto")``
* ``base`` / ``base-0.6b`` (reusable clone prompt path):
  ``prompt = model.create_voice_clone_prompt(ref_audio, ref_text)`` built ONCE,
  then ``generate_voice_clone(text=[...], language="Auto",
  voice_clone_prompt=prompt)`` — the wrapper broadcasts the 1-item prompt
  over the text list (qwen-tts 0.1.1 ``_ensure_list`` semantics).

Ladder policy (binding): batch sizes 1, 2, 4, 8 per family; the FIRST real
failure (exception or any insane output) is recorded verbatim and the ladder
STOPS for that family — no retries at higher sizes; ``meta.per_family_max_stable``
reports the maximum validated batch size. Failures are evidence, never hidden.

Methodology (mirrors scripts/benchmark.py where applicable): one discarded
warmup generation per family (kernel warmup), ``torch.manual_seed(SEED)``
before every batch call (the official API's only seed surface is the global
torch RNG), ``max_new_tokens=512`` latency guardrail, waveforms validated
finite + non-silent (RMS >= 1e-3) + sample rate 24000 per item, wall time
per official-API call, torch peak allocated/reserved reset per call, and
``rocm-smi`` VRAM deltas sampled per family when parseable.

Usage::

    .venv/bin/python scripts/benchmark_batch.py                  # all 5 families, B=1,2,4,8
    .venv/bin/python scripts/benchmark_batch.py --families custom-voice,base \\
        --batch-sizes 1,4 --json-out evidence/batch-inference-smoke.json

Outputs: markdown-ready rows on stdout AND a JSON document (default
``evidence/batch-inference.json``): ``{meta:{...per_family_max_stable},
results:[per-cell records]}``. Importing this module is lightweight (stdlib
only); torch/qwen_tts/loader import lazily inside functions.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

#: Fixed manifest: 8 distinct short zh sentences (deterministic order; the
#: first B of them are used at batch size B so bigger batches strictly extend
#: smaller ones — same-prefix property keeps ladder cells comparable).
MANIFEST: list[str] = [
    "今天的天气真不错，适合去公园散步。",
    "书桌上放着一杯热茶和几本旧书。",
    "窗外的鸟叫声让人心情愉快。",
    "请帮我查一下明天下午的会议安排。",
    "这条街的夜晚总是很热闹。",
    "孩子们在操场上追逐嬉戏。",
    "咖啡的香气弥漫在整个房间里。",
    "远处的山峦在晨雾中若隐若现。",
]

#: VoiceDesign instruction (same style as the VoiceDesign GPU suite).
INSTRUCT = "用温和愉悦的语气说"

#: Global torch seed set before every batch call (official API's seed surface).
SEED = 20260924

MAX_NEW_TOKENS = 512

MD_HEADER = "| family | alias | B | wall s | audio s/item | total audio s | audio/wall s | RTF | peak alloc GiB | all sane |"
MD_SEP = "|---|---|---|---|---|---|---|---|---|---|"

FAMILIES: dict[str, dict[str, str]] = {
    "custom-voice-0.6b": {"alias": "custom-voice-0.6b", "kind": "custom_voice"},
    "custom-voice": {"alias": "custom-voice", "kind": "custom_voice"},
    "voice-design": {"alias": "voice-design", "kind": "voice_design"},
    "base-0.6b": {"alias": "base-0.6b", "kind": "clone_prompt"},
    "base": {"alias": "base", "kind": "clone_prompt"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Official-API batch-inference benchmark (v0.2.1 Task 3)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"comma-separated family keys (default: all; known: {list(FAMILIES)})")
    ap.add_argument("--batch-sizes", default="1,2,4,8",
                    help="ascending batch-size ladder (default: 1,2,4,8)")
    ap.add_argument("--json-out", default="evidence/batch-inference.json",
                    help="machine-readable output path")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="torch.manual_seed value set before every batch call")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="official latency guardrail kwarg (default: 512)")
    return ap.parse_args(argv)


def _vram_used_gb() -> float | None:
    """Total VRAM used in GiB via rocm-smi, or None when not parseable."""
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=20,
                             check=False).stdout
        vals = [int(line.split("(B):")[1].strip()) for line in out.splitlines()
                if "VRAM Total Used Memory" in line]
        return round(vals[0] / 2**30, 2) if vals else None
    except Exception:  # noqa: BLE001 - VRAM probe never blocks the bench
        return None


def _warmup(model, alias: str) -> None:
    """One discarded short generation (kernel + code-path warmup)."""
    import torch

    from qwen3_tts_rocm import testing

    torch.manual_seed(SEED)
    method = "generate_voice_clone" if alias.startswith("base") else (
        "generate_voice_design" if alias.startswith("voice-design") else "generate_custom_voice")
    if method == "generate_custom_voice":
        wavs, sr = model.generate_custom_voice(
            text=MANIFEST[0], language="Auto",
            speaker=model.get_supported_speakers()[0],
            max_new_tokens=MAX_NEW_TOKENS)
    elif method == "generate_voice_design":
        wavs, sr = model.generate_voice_design(
            text=MANIFEST[0], instruct=INSTRUCT, language="Auto",
            max_new_tokens=MAX_NEW_TOKENS)
    else:
        prompt = model.create_voice_clone_prompt(
            ref_audio=_ref_audio(), ref_text=_REF_TEXT)
        wavs, sr = model.generate_voice_clone(
            text=MANIFEST[0], language="Auto", voice_clone_prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS)
    testing.assert_wav_sane(wavs[0], sr_expected=sr)


def _ref_audio():
    """The license-clean synthetic reference asset used across the clone suites."""
    from importlib.resources import files

    import numpy as np
    import soundfile as sf

    path = str(files("qwen3_tts_rocm") / "demo" / "assets" / "ref_en.wav")
    wav, sr = sf.read(path, dtype="float32")
    return np.ascontiguousarray(wav), sr


_REF_TEXT = "This tiny synthetic voice was cloned for automated testing."


def _run_batch(model, family: str, kind: str, b: int, args) -> tuple[list, int]:
    """One official-API batched call; returns (wavs, sr)."""
    import torch

    torch.manual_seed(args.seed)
    texts = MANIFEST[:b]
    if kind == "custom_voice":
        return model.generate_custom_voice(
            text=texts, language="Auto",
            speaker=model.get_supported_speakers()[0],
            max_new_tokens=args.max_new_tokens)
    if kind == "voice_design":
        return model.generate_voice_design(
            text=texts, instruct=INSTRUCT, language="Auto",
            max_new_tokens=args.max_new_tokens)
    if kind == "clone_prompt":
        prompt = model.create_voice_clone_prompt(
            ref_audio=_ref_audio(), ref_text=_REF_TEXT)
        return model.generate_voice_clone(
            text=texts, language="Auto", voice_clone_prompt=prompt,
            max_new_tokens=args.max_new_tokens)
    raise ValueError(f"unknown kind {kind} for family {family}")


def _cell(model, family: str, alias: str, kind: str, b: int, args) -> dict:
    """Measure one ladder cell; raise on any real failure (caller records)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    wavs, sr = _run_batch(model, family, kind, b, args)
    wall = time.perf_counter() - t0

    import numpy as np

    durs = [float(w.shape[-1]) / sr for w in wavs]
    finite = all(bool(np.isfinite(w).all()) for w in wavs)
    non_silent = all(float((w.astype("float32") ** 2).mean() ** 0.5) >= 1e-3 for w in wavs)
    total_audio = sum(durs)
    ok = (len(wavs) == b and sr == 24000 and finite and non_silent
          and all(0.5 <= d <= 60.0 for d in durs))
    if not ok:
        maxabs = max(float(abs(w).max()) for w in wavs)
        raise RuntimeError(
            f"insane batch output: n={len(wavs)}/{b} sr={sr} finite={finite} "
            f"non_silent={non_silent} durs={[round(d,2) for d in durs]} maxabs={maxabs:.4f}")
    return {
        "family": family, "alias": alias, "kind": kind, "batch_size": b,
        "prompt_char_lengths": [len(t) for t in MANIFEST[:b]],
        "wall_seconds": round(wall, 2),
        "audio_seconds_per_item": [round(d, 2) for d in durs],
        "total_audio_seconds": round(total_audio, 2),
        "audio_per_wall_second": round(total_audio / wall, 2),
        "rtf": round(wall / total_audio, 2),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available() else None,
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if torch.cuda.is_available() else None,
        "sample_rate": sr, "ok": True,
    }


def md_row(r: dict) -> str:
    return (f"| {r['family']} | {r['alias']} | {r['batch_size']} | {r['wall_seconds']} | "
            f"{r['audio_seconds_per_item']} | {r['total_audio_seconds']} | "
            f"{r['audio_per_wall_second']} | {r['rtf']} | {r['peak_alloc_gb']} | yes |")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    if unknown or not families or not sizes or sizes != sorted(sizes):
        print(f"ERROR: bad families {unknown} or batch sizes {sizes}", file=sys.stderr)
        return 2

    import torch

    from qwen3_tts_rocm import loader, models

    for f in families:
        if FAMILIES[f]["alias"] not in models.ALIASES:
            print(f"ERROR: alias {FAMILIES[f]['alias']} not in registry", file=sys.stderr)
            return 2

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": None, "torch": torch.__version__, "hip": str(torch.version.hip),
        "git_head": _git_head(), "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "manifest": MANIFEST,
        "batch_sizes_requested": sizes, "per_family_max_stable": {},
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        meta["gpu"] = f"{p.name} ({getattr(p, 'gcnArchName', '?')})"

    print("== qwen3-tts-rocm official-API batch benchmark ==")
    print(f"# date={meta['date']} host={meta['host']}")
    print(f"# gpu={meta['gpu']} torch={meta['torch']} (HIP {meta['hip']})")
    print(f"# args: families={families} batch_sizes={sizes} seed={args.seed} "
          f"max_new_tokens={args.max_new_tokens}")

    results: list[dict] = []
    rows: list[str] = []
    for family in families:
        alias, kind = FAMILIES[family]["alias"], FAMILIES[family]["kind"]
        print(f"[batch] loading alias={alias} ...", flush=True)
        vram_before = _vram_used_gb()
        t0 = time.perf_counter()
        model = loader.load(alias)
        load_seconds = time.perf_counter() - t0
        print(f"[batch] loaded alias={alias} took={load_seconds:.1f}s "
              f"vram_before={vram_before} GiB", flush=True)
        max_stable = 0
        try:
            _warmup(model, alias)
            print(f"[batch] warmup alias={alias} done (discarded)", flush=True)
            for b in sizes:
                try:
                    rec = _cell(model, family, alias, kind, b, args)
                except Exception:  # noqa: BLE001 - ladder policy: any real failure is recorded verbatim
                    tb = traceback.format_exc()
                    print(f"[batch] FAIL family={family} B={b} — ladder stops here "
                          f"(verbatim traceback follows)\n{tb}", flush=True)
                    results.append({
                        "family": family, "alias": alias, "kind": kind,
                        "batch_size": b, "ok": False, "error": tb.strip().splitlines()[-1],
                        "error_verbatim": tb,
                    })
                    break
                rec["load_seconds"] = round(load_seconds, 2)
                rec["vram_used_gb_before_family"] = vram_before
                rec["vram_used_gb_after_family"] = None  # filled after unload
                results.append(rec)
                rows.append(md_row(rec))
                max_stable = b
                print(f"[batch] OK family={family} B={b} wall={rec['wall_seconds']}s "
                      f"audio_total={rec['total_audio_seconds']}s rtf={rec['rtf']} "
                      f"peak_alloc={rec['peak_alloc_gb']} GiB", flush=True)
        finally:
            loader.unload(model)
            vram_after = _vram_used_gb()
            for r in results:
                if r.get("family") == family and r.get("vram_used_gb_after_family") is None:
                    r["vram_used_gb_after_family"] = vram_after
        meta["per_family_max_stable"][family] = max_stable
        print(f"[batch] family={family} max_stable_batch={max_stable} "
              f"vram_after={vram_after} GiB", flush=True)

    meta["results_count"] = len(results)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[batch] wrote {args.json_out}", flush=True)

    print(MD_HEADER)
    print(MD_SEP)
    for row in rows:
        print(row)
    failed = [r for r in results if not r.get("ok")]
    if failed:
        print(f"# NOTE: {len(failed)} failed ladder cell(s) recorded verbatim above/in JSON")
    # A family failing even at B=1 means the official single-sample path broke:
    # that is a real capability regression, not a batch boundary.
    broken = [f for f, m in meta["per_family_max_stable"].items() if m < 1]
    if broken:
        print(f"ERROR: families failing at B=1: {broken}", file=sys.stderr)
        return 1
    print("BATCH-BENCHMARK-OK", flush=True)
    return 0


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance probe never blocks the bench
        return None


if __name__ == "__main__":
    sys.exit(main())
