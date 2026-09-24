#!/usr/bin/env python3
"""Controlled-reproducibility RTF benchmark v2 (v0.2.1 Task 4).

Methodology v2 — what changed vs ``scripts/benchmark.py`` (v1) and why:

* n=5 measured runs per cell (v1 default was 2) — enough for median/min/
  max/mean/stdev; p10/p90 are additionally reported as INCLUSIVE linear
  interpolations of a 5-point sample, i.e. rough order statistics, not
  confident tail estimates (stated in docs/benchmarks-v2.md).
* Fixed seeds per measured run: ``torch.manual_seed(seed_base + k)`` before
  every generation (the official API's only seed surface is the global torch
  RNG). Same seed list every session => the token streams are reproducible;
  different seeds ACROSS the 5 runs keep per-run generation work realistic,
  so run-to-run spread measures real variance, not just scheduler noise.
* Explicit phase separation: model LOAD time (cold, timed), WARMUP time
  (the first post-load generation — same cell as cn/short, timed and
  RECORDED but excluded from measured statistics), and the measured
  generation runs (warm model). docs/benchmarks-v2.md maps which number is
  which; nothing is silently discarded.
* Environment bookends: ``rocm-smi`` VRAM + temperature/clock snapshot
  before load and after the last generation of every alias (raw lines kept
  verbatim in the JSON — sensor availability varies, nothing is invented).
* Coverage: ALL five registered TTS aliases (1.7B + 0.6B per family) —
  never collapsed into one "Radeon performance" number.
* Backward compatibility: this is a NEW script; ``scripts/benchmark.py``
  (v1, and the CI ``full-weekly`` replication step) is untouched.

Per-run record: seed, wall seconds, audio seconds, RTF (wall/audio, lower
is better — repo-binding definition), plus per-alias torch peak
allocated/reserved and the rocm-smi bookends.

Usage::

    .venv/bin/python scripts/benchmark_v2.py                     # all 5 aliases, n=5
    .venv/bin/python scripts/benchmark_v2.py --aliases base --runs 3 \
        --json-out /tmp/v2-smoke.json

Importing this module is lightweight (stdlib only); torch and the loader
import lazily inside functions, exactly like benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# v1 script is the source of the FIXED PROMPT MANIFEST (TEXTS), the
# official-entry-point selector (build_call), and the peak-memory helpers —
# importing them keeps v1 and v2 cells directly comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    LANG_KEYS,
    TEXTS,
    build_call,
    peak_alloc_gb,
    reset_peak_gpu_memory,
)

ALIASES_V2 = "custom-voice,custom-voice-0.6b,voice-design,base,base-0.6b"
RUNS_V2 = 5
SEED_BASE = 20260924
MAX_NEW_TOKENS = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Controlled-reproducibility RTF benchmark v2 (v0.2.1 Task 4)")
    ap.add_argument("--aliases", default=ALIASES_V2,
                    help="comma-separated registry aliases (default: all five)")
    ap.add_argument("--runs", type=int, default=RUNS_V2,
                    help="measured runs per cell (default: 5)")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE,
                    help="torch.manual_seed base; run k uses seed_base+k")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="official latency guardrail kwarg (default: 512)")
    ap.add_argument("--json-out", default="evidence/benchmark-v2.json",
                    help="machine-readable output path")
    return ap.parse_args(argv)


def cell_stats_v2(values: list[float]) -> dict:
    """n=5 statistics for one cell's runs (pure; unit-tested).

    p10/p90 use statistics.quantiles(..., n=100, method="inclusive") —
    linear interpolation over a 5-point sample: order-statistic hints,
    NOT confident tail estimates (caveat lives in docs/benchmarks-v2.md).
    """
    if not values:
        return {}
    if len(values) == 1:
        v = round(values[0], 2)
        return {"n": 1, "median": v, "min": v, "max": v, "mean": v,
                "stdev": 0.0, "p10": v, "p90": v}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "n": len(values),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "p10": round(q[9], 2),
        "p90": round(q[89], 2),
    }


def _rocm_smi_bookend() -> dict:
    """Raw rocm-smi VRAM/temperature/clock snapshot; verbatim, never parsed
    into invented numbers (sensor availability varies by driver)."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(["rocm-smi", *args], capture_output=True,
                                  text=True, timeout=20, check=False).stdout.strip()
        except Exception:  # noqa: BLE001 - environment probe never blocks the bench
            return ""
    return {"vram": run("--showmeminfo", "vram"),
            "temp_clocks": run("--showtemp", "--showclocks")}


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance probe never blocks the bench
        return None


def _generate_once(model, alias: str, lang_key: str, length_key: str, seed: int,
                   max_new_tokens: int) -> dict:
    """One seeded official-API generation; returns the per-run record."""
    import torch

    from qwen3_tts_rocm import testing

    torch.manual_seed(seed)
    method, kwargs = build_call(model, alias, TEXTS[lang_key][length_key],
                                language=LANG_KEYS[lang_key])
    t0 = time.perf_counter()
    # Same official-neutral sampling bindings as v1 run_cell (comparability):
    # do_sample=True, temperature=1.0 explicit; other knobs = model defaults.
    wavs, sr = getattr(model, method)(**kwargs, do_sample=True, temperature=1.0,
                                      max_new_tokens=max_new_tokens)
    wall = time.perf_counter() - t0
    testing.assert_wav_sane(wavs[0], sr_expected=sr)
    audio = float(wavs[0].shape[-1]) / sr
    return {"seed": seed, "wall_seconds": round(wall, 2),
            "audio_seconds": round(audio, 2),
            "rtf": round(wall / audio, 2)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    if not aliases or args.runs < 1:
        print("ERROR: need >=1 alias and >=1 run", file=sys.stderr)
        return 2

    import torch

    from qwen3_tts_rocm import loader, models

    unknown = [a for a in aliases if a not in models.ALIASES]
    if unknown:
        print(f"ERROR: unknown alias(es) {unknown}; known: {list(models.ALIASES)}",
              file=sys.stderr)
        return 2

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": None, "torch": torch.__version__, "hip": str(torch.version.hip),
        "git_head": _git_head(),
        "methodology": f"v2 (n={args.runs}, seeded runs, load/warmup/gen separated)",
        "seed_base": args.seed_base, "max_new_tokens": args.max_new_tokens,
        "runs_per_cell": args.runs,
        "prompt_manifest": {lk: dict(tv) for lk, tv in TEXTS.items()},
        "smi_bookends_note": "rocm-smi raw output kept verbatim; sensors vary by driver",
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        meta["gpu"] = f"{p.name} ({getattr(p, 'gcnArchName', '?')})"

    print("== qwen3-tts-rocm RTF benchmark v2 ==")
    print(f"# date={meta['date']} host={meta['host']}")
    print(f"# gpu={meta['gpu']} torch={meta['torch']} (HIP {meta['hip']})")
    print(f"# aliases={aliases} runs={args.runs} seed_base={args.seed_base} "
          f"max_new_tokens={args.max_new_tokens}")

    seed_k = 0
    per_alias: dict[str, dict] = {}
    failed: list[dict] = []
    for alias in aliases:
        print(f"[v2] alias={alias}: smi bookend BEFORE LOAD", flush=True)
        smi_before = _rocm_smi_bookend()
        t0 = time.perf_counter()
        model = loader.load(alias)
        load_seconds = time.perf_counter() - t0
        reset_peak_gpu_memory()
        print(f"[v2] alias={alias} loaded in {load_seconds:.1f}s (COLD-START load time)",
              flush=True)

        block: dict = {"load_seconds": round(load_seconds, 2), "cells": [],
                       "smi_before": smi_before}
        try:
            # Warmup = first post-load generation (cn/short), timed and
            # RECORDED, excluded from measured statistics. Unlike v1's
            # deliberately UNCAPPED warmup (official 2048-token default; a
            # degenerate base run can burn ~23 min), v2 warmup keeps the
            # 512-token guardrail — the warmup path is identical, only the
            # runaway ceiling differs (documented in docs/benchmarks-v2.md).
            seed_k += 1
            warm = _generate_once(model, alias, "cn", "short",
                                  args.seed_base + seed_k, args.max_new_tokens)
            block["warmup"] = warm
            print(f"[v2] alias={alias} warmup (cn/short, recorded, excluded from "
                  f"stats): wall={warm['wall_seconds']}s audio={warm['audio_seconds']}s",
                  flush=True)

            for lang_key in ("cn", "en"):
                for length_key in ("short", "medium"):
                    runs = []
                    for _ in range(args.runs):
                        seed_k += 1
                        runs.append(_generate_once(
                            model, alias, lang_key, length_key,
                            args.seed_base + seed_k, args.max_new_tokens))
                    cell = {
                        "lang": lang_key, "len": length_key, "runs": runs,
                        "wall_stats": cell_stats_v2([r["wall_seconds"] for r in runs]),
                        "audio_stats": cell_stats_v2([r["audio_seconds"] for r in runs]),
                        "rtf_stats": cell_stats_v2([r["rtf"] for r in runs]),
                    }
                    block["cells"].append(cell)
                    s = cell["rtf_stats"]
                    print(f"[v2] alias={alias} {lang_key}/{length_key}: RTF median="
                          f"{s['median']} min={s['min']} max={s['max']} mean={s['mean']} "
                          f"stdev={s['stdev']} p10={s['p10']} p90={s['p90']} "
                          f"(n={s['n']})", flush=True)
        except Exception:  # noqa: BLE001 - a real failure ends the alias, recorded verbatim
            tb = traceback.format_exc()
            print(f"[v2] FAIL alias={alias} — verbatim traceback follows\n{tb}",
                  flush=True)
            failed.append({"alias": alias, "error": tb.strip().splitlines()[-1],
                           "error_verbatim": tb})
        finally:
            block["peak_alloc_gb"] = peak_alloc_gb()
            block["peak_reserved_gb"] = None
            if torch.cuda.is_available():
                block["peak_reserved_gb"] = round(
                    torch.cuda.max_memory_reserved() / 2**30, 2)
            block["smi_after"] = _rocm_smi_bookend()
            per_alias[alias] = block
            print(f"[v2] alias={alias}: smi bookend AFTER; peak_alloc="
                  f"{block['peak_alloc_gb']} GiB peak_reserved="
                  f"{block['peak_reserved_gb']} GiB — unloading", flush=True)
            loader.unload(model)

    doc = {"meta": meta, "per_alias": per_alias}
    if failed:
        doc["failures"] = failed
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[v2] wrote {out}", flush=True)

    print("| alias | lang | len | n | RTF median | min | max | mean | stdev | p10 | p90 | load s | warmup s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for alias, block in per_alias.items():
        for cell in block.get("cells", []):
            s = cell["rtf_stats"]
            print(f"| {alias} | {cell['lang']} | {cell['len']} | {s['n']} | {s['median']} | "
                  f"{s['min']} | {s['max']} | {s['mean']} | {s['stdev']} | {s['p10']} | "
                  f"{s['p90']} | {block['load_seconds']} | {block['warmup']['wall_seconds']} |")

    if failed:
        print(f"ERROR: {len(failed)} alias(es) failed: {[f['alias'] for f in failed]}",
              file=sys.stderr)
        return 1
    print("BENCHMARK-V2-OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
