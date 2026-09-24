#!/usr/bin/env python3
"""Soak test for qwen3-tts-rocm on gfx1151 (v0.2.1 Task 5B).

Two modes (run separately or back-to-back via --mode):

* ``resident`` — ONE model stays loaded; deterministic requests synthesize
  repeatedly for a bounded session (--minutes, default 30). Every request
  appends a JSONL telemetry record: success/failure counts, wall time, GPU
  memory (torch peak alloc/reserved since last record + rocm-smi VRAM when
  parseable), process RSS, generated audio duration, temperature (raw
  rocm-smi line, kept verbatim — sensor availability varies), cumulative
  failure count. Memory-growth analysis = comparing first vs last records
  (the verifier does this independently; this script only records).
* ``recycle`` — load → generate → unload cycles (--cycles, default 10),
  testing allocator cleanup: per-cycle load seconds, generate wall/audio,
  RSS after unload, torch peak. Growth in RSS-after-unload across cycles is
  the allocator-leak signal (recorded, not claimed).

Deterministic manifest: requests cycle a fixed list of zh/en texts with a
fixed seed per request index (seed_base + i), so any two runs issue
byte-identical work. Correctness gate per request: waveform finite,
non-silent (RMS >= 1e-3), 24 kHz, duration in [0.5, 60] s — a failure
appends the verbatim error and, per the binding policy, exits NONZERO at
the end of the run (the session continues to completion first so the
telemetry stays useful; --fail-fast stops immediately instead).

NO "memory leak free" claim is made anywhere unless a measured baseline
supports it; this script records, humans interpret.

Usage::

    .venv/bin/python scripts/soak_test.py --mode resident --minutes 30 \
        --alias custom-voice --jsonl-out evidence/soak-resident.jsonl
    .venv/bin/python scripts/soak_test.py --mode recycle --cycles 10 \
        --alias custom-voice --jsonl-out evidence/soak-recycle.jsonl

Importing this module is lightweight (stdlib only); torch imports lazily.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Official language names for the manifest keys. Deliberately does NOT
#: reuse v1 benchmark.py's LANG_KEYS (keyed "cn"): soak attempt 1 archived
#: the same KeyError('zh') bug class as the ladder's first attempt — every
#: zh request failed while every en request passed (2026-09-24).
LANG_NAMES: dict[str, str] = {"zh": "Chinese", "en": "English"}

#: Deterministic request manifest — zh/en alternating, short sentences
#: (512-token guardrail keeps every request bounded; soak measures stability,
#: not long-form capability — that is longtext_ladder.py's job).
REQUESTS: list[tuple[str, str]] = [
    ("zh", "今天的天气真不错，适合去公园散步。"),
    ("en", "The weather is lovely today, perfect for a walk in the park."),
    ("zh", "书桌上放着一杯热茶和几本旧书。"),
    ("en", "A warm cup of tea and a few old books sit on the desk."),
    ("zh", "窗外的鸟叫声让人心情愉快。"),
    ("en", "The birds outside the window lift everyone's mood."),
]

MAX_NEW_TOKENS = 512
SEED_BASE = 20260925


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Soak test (v0.2.1 Task 5B)")
    ap.add_argument("--mode", choices=("resident", "recycle"), default="resident")
    ap.add_argument("--alias", default="custom-voice")
    ap.add_argument("--speaker-index", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="resident mode: session length in minutes")
    ap.add_argument("--cycles", type=int, default=10,
                    help="recycle mode: load→generate→unload repetitions")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort on first correctness failure (default: finish the "
                         "session, record verbatim, exit nonzero)")
    ap.add_argument("--jsonl-out", default="evidence/soak.jsonl")
    return ap.parse_args(argv)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return -1.0


def _rocm_smi() -> dict:
    def run(*flags: str) -> str:
        try:
            return subprocess.run(["rocm-smi", *flags], capture_output=True,
                                  text=True, timeout=20, check=False).stdout.strip()
        except Exception:  # noqa: BLE001 - probe never blocks the soak
            return ""
    vram = run("--showmeminfo", "vram")
    used = None
    for line in vram.splitlines():
        if "VRAM Total Used Memory" in line:
            used = round(int(line.split("(B):")[1].strip()) / 2**30, 2)
    return {"vram_used_gb": used, "temp_raw": run("--showtemp")}


def _generate(model, alias: str, lang: str, text: str, seed: int, args) -> dict:
    import torch


    torch.manual_seed(seed)
    family = alias.split("-0.6b")[0]
    kwargs: dict = {"text": text, "language": LANG_NAMES[lang]}
    if family == "custom-voice":
        kwargs["speaker"] = model.get_supported_speakers()[args.speaker_index]
        method = "generate_custom_voice"
    elif family == "voice-design":
        from benchmark import VOICE_DESIGN_INSTRUCT
        kwargs["instruct"] = VOICE_DESIGN_INSTRUCT
        method = "generate_voice_design"
    elif family == "base":
        from benchmark import BASE_REF_TEXT, _load_base_ref_audio
        wav, sr = _load_base_ref_audio()
        kwargs["ref_audio"] = (wav, sr)
        kwargs["ref_text"] = BASE_REF_TEXT
        method = "generate_voice_clone"
    else:
        raise SystemExit(f"unknown alias family for {alias!r}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    wavs, sr_out = getattr(model, method)(**kwargs, max_new_tokens=args.max_new_tokens)
    wall = time.perf_counter() - t0
    wav = wavs[0]
    import numpy as np

    finite = bool(np.isfinite(wav).all())
    rms = float((wav.astype("float32") ** 2).mean() ** 0.5)
    dur = float(wav.shape[-1]) / sr_out
    sane = finite and rms >= 1e-3 and sr_out == 24000 and 0.5 <= dur <= 60.0
    return {"ok": bool(sane), "wall_seconds": round(wall, 2),
            "audio_seconds": round(dur, 2), "rtf": round(wall / dur, 2),
            "rms": round(rms, 4), "sr": int(sr_out),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else None,
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
            if torch.cuda.is_available() else None}


def _meta(args) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    return {"kind": "soak_meta", "date": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"), "host": platform.node(),
            "mode": args.mode, "alias": args.alias, "git_head": head,
            "seed_base": args.seed_base, "max_new_tokens": args.max_new_tokens,
            "manifest": [{"lang": l, "text": t} for l, t in REQUESTS]}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from qwen3_tts_rocm import loader

    out = Path(args.jsonl_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", encoding="utf-8")  # noqa: SIM115 - held open for the whole session, closed in finally
    meta = _meta(args)
    meta["torch"] = torch.__version__
    meta["hip"] = str(torch.version.hip)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        meta["gpu"] = f"{p.name} ({getattr(p, 'gcnArchName', '?')})"
    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[soak] mode={args.mode} alias={args.alias} -> {out}", flush=True)

    successes = failures = 0
    cumulative_fail = 0
    deadline = time.monotonic() + args.minutes * 60
    i = 0
    session_start = time.monotonic()
    stop_reason = None

    def gen_once(model) -> dict:
        nonlocal successes, failures, cumulative_fail, i
        lang, text = REQUESTS[i % len(REQUESTS)]
        seed = args.seed_base + i
        i += 1
        try:
            rec = _generate(model, args.alias, lang, text, seed, args)
            if rec["ok"]:
                successes += 1
            else:
                failures += 1
                cumulative_fail += 1
                rec["error"] = "sanity gate failed"
        except Exception:  # noqa: BLE001 - verbatim failure recording
            tb = traceback.format_exc()
            failures += 1
            cumulative_fail += 1
            rec = {"ok": False, "error": tb.strip().splitlines()[-1],
                   "error_verbatim": tb}
        rec.update({"request_index": i, "lang": lang,
                    "elapsed_seconds": round(time.monotonic() - session_start, 1),
                    "successes": successes, "failures": failures,
                    "cumulative_failures": cumulative_fail,
                    "rss_gb": _rss_gb()})
        return rec

    try:
        if args.mode == "resident":
            model = loader.load(args.alias)
            print(f"[soak] resident model loaded; session length {args.minutes} min",
                  flush=True)
            while True:
                if time.monotonic() >= deadline:
                    stop_reason = "deadline"
                    break
                rec = gen_once(model)
                rec["rocm"] = _rocm_smi() if rec["request_index"] % 5 == 0 else None
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                status = "OK " if rec["ok"] else "FAIL"
                print(f"[soak] req#{rec['request_index']} {status} wall="
                      f"{rec.get('wall_seconds')}s audio={rec.get('audio_seconds')}s "
                      f"rss={rec['rss_gb']}GiB peak={rec.get('peak_alloc_gb')}GiB "
                      f"cum_fail={cumulative_fail}", flush=True)
                if not rec["ok"] and args.fail_fast:
                    stop_reason = "fail_fast"
                    break
            loader.unload(model)
        else:  # recycle
            for c in range(1, args.cycles + 1):
                t0 = time.perf_counter()
                model = loader.load(args.alias)
                load_s = time.perf_counter() - t0
                rec = gen_once(model)
                loader.unload(model)
                rec["cycle"] = c
                rec["load_seconds"] = round(load_s, 2)
                rec["rss_gb_after_unload"] = _rss_gb()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[soak] cycle#{c} {'OK' if rec['ok'] else 'FAIL'} load="
                      f"{rec['load_seconds']}s wall={rec.get('wall_seconds')}s "
                      f"rss_after_unload={rec['rss_gb_after_unload']}GiB", flush=True)
                if not rec["ok"] and args.fail_fast:
                    stop_reason = "fail_fast"
                    break
            stop_reason = stop_reason or "cycles_completed"
    finally:
        f.write(json.dumps({"kind": "soak_summary", "stop_reason": stop_reason,
                            "total_requests": successes + failures,
                            "successes": successes, "failures": failures,
                            "session_minutes": round((time.monotonic() - session_start) / 60, 2),
                            "exit_code": 0 if failures == 0 else 1},
                           ensure_ascii=False) + "\n")
        f.close()

    if failures:
        print(f"SOAK-FAILED: {failures} correctness failure(s) — see {out}",
              file=sys.stderr, flush=True)
        return 1
    print("SOAK-OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
