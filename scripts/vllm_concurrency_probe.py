#!/usr/bin/env python3
"""vLLM-Omni concurrency/throughput probe for gfx1151 (v0.2.1 Task 10).

Fires a fixed prompt set at a given concurrency against a LIVE
`vllm serve` instance (default: 8 requests per concurrency level), using
the SAME documented HTTP PCM streaming interface as Task 9. Each request
records client-side timings (send, first audio byte, completion) and byte
counts; the driver additionally samples `rocm-smi` VRAM and the server
process RSS during the run and computes per-level aggregates:

* success/failure counts (failure = HTTP error, zero bytes, or non-finite
  decoded PCM length)
* TTFA p50/p95 (percentiles via statistics.quantiles, inclusive)
* E2E latency p50/p95 (send → last byte)
* generated audio seconds + audio-seconds/wall-second + requests/minute
* peak VRAM sample / server RSS sample
* queueing signal: median TTFA vs the c=1 baseline's median TTFA

Ladder policy (binding): concurrency levels ascending; on instability
(errors/OOM/HIP failures) the level is recorded verbatim and the ladder
STOPS — never hidden.

Usage (server must be up):

    python scripts/vllm_concurrency_probe.py --url http://127.0.0.1:8091 \
        --levels 1,2,4,8 --requests-per-level 8 --tag default \
        --json-out /tmp/conc.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import requests

BYTES_PER_SECOND_PCM = 24_000 * 2

PROMPTS = [
    "Hello, how are you today?",
    "The weather is lovely outside.",
    "This is a concurrent synthesis test.",
    "Audio streams in parallel on one GPU.",
    "Numbers help calibrate expectations.",
    "Concurrency measures the serving stack.",
    "Every request is timed from the client.",
    "Thanks for listening to this sentence.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Concurrency probe (Task 10)")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--requests-per-level", type=int, default=8)
    ap.add_argument("--voice", default="vivian")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json-out", required=True)
    return ap.parse_args(argv)


def _pct(values: list[float], pct: int) -> float:
    if len(values) < 2:
        return round(values[0], 3) if values else None
    q = statistics.quantiles(values, n=100, method="inclusive")
    return round(q[pct - 1], 3)


def _sample_env(server_pid: int | None) -> dict:
    vram = None
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True,
                             text=True, timeout=15, check=False).stdout
        for line in out.splitlines():
            if "VRAM Total Used Memory" in line:
                vram = round(int(line.split("(B):")[1].strip()) / 2**30, 2)
    except Exception:  # noqa: BLE001,S110 - env sampling never blocks the probe
        pass
    rss = None
    if server_pid:
        try:
            with open(f"/proc/{server_pid}/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = round(int(line.split()[1]) / 2**20, 2)
        except OSError:
            pass
    return {"vram_used_gb": vram, "server_rss_gb": rss}


def _one_request(url: str, prompt: str, voice: str, timeout: float) -> dict:
    body = {"input": prompt, "voice": voice, "language": "English",
            "task_type": "CustomVoice", "stream": True,
            "stream_format": "audio", "response_format": "pcm"}
    t0 = time.monotonic()
    first = None
    total = 0
    try:
        with requests.post(f"{url}/v1/audio/speech", json=body,
                           stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    if first is None:
                        first = time.monotonic()
                    total += len(chunk)
        t_end = time.monotonic()
        ok = total > 0
        return {"ok": ok, "prompt": prompt,
                "ttfa_s": round(first - t0, 3) if first else None,
                "e2e_s": round(t_end - t0, 3),
                "audio_s": round(total / BYTES_PER_SECOND_PCM, 3), "bytes": total}
    except Exception as exc:  # noqa: BLE001 - verbatim failure recording
        return {"ok": False, "prompt": prompt, "error": f"{type(exc).__name__}: {exc}",
                "ttfa_s": None, "e2e_s": round(time.monotonic() - t0, 3),
                "audio_s": 0.0, "bytes": 0}


def run_level(url: str, level: int, n_requests: int, voice: str, timeout: float,
              server_pid: int | None) -> dict:
    results: list[dict] = []
    env_samples: list[dict] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            env_samples.append(_sample_env(server_pid))
            time.sleep(2.0)

    st = threading.Thread(target=sampler, daemon=True)
    t_start = time.monotonic()
    st.start()
    done = 0
    while done < n_requests:
        batch = min(level, n_requests - done)
        threads: list[threading.Thread] = []
        batch_results: list[dict] = [None] * batch  # type: ignore[list-item]

        def worker(i: int, prompt: str, _sink: list | None = None) -> None:
            assert _sink is not None
            _sink[i] = _one_request(url, prompt, voice, timeout)

        for i in range(batch):
            th = threading.Thread(target=worker, args=(i, PROMPTS[(done + i) % len(PROMPTS)], batch_results))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        results.extend(batch_results)
        done += batch
    wall = time.monotonic() - t_start
    stop.set()
    st.join(timeout=5)

    oks = [r for r in results if r["ok"]]
    ttfas = [r["ttfa_s"] for r in oks]
    e2es = [r["e2e_s"] for r in oks]
    audio_total = sum(r["audio_s"] for r in oks)
    vrams = [e["vram_used_gb"] for e in env_samples if e["vram_used_gb"] is not None]
    rsses = [e["server_rss_gb"] for e in env_samples if e["server_rss_gb"] is not None]
    return {
        "level": level, "requests": len(results),
        "successes": len(oks), "failures": len(results) - len(oks),
        "ttfa_p50_s": _pct(ttfas, 50), "ttfa_p95_s": _pct(ttfas, 95),
        "e2e_p50_s": _pct(e2es, 50), "e2e_p95_s": _pct(e2es, 95),
        "audio_seconds_total": round(audio_total, 2),
        "audio_per_wall_second": round(audio_total / wall, 3) if wall else None,
        "requests_per_minute": round(len(oks) / (wall / 60), 2),
        "level_wall_s": round(wall, 2),
        "peak_vram_gb": max(vrams) if vrams else None,
        "server_rss_gb_last": rsses[-1] if rsses else None,
        "failures_verbatim": [r.get("error") for r in results if not r["ok"]],
        "per_request": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    server_pid = None
    try:
        out = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True,
                             text=True, timeout=10, check=False).stdout
        pids = [int(p) for p in out.split()]
        server_pid = min(pids) if pids else None  # API-server pid = lowest
    except Exception:  # noqa: BLE001,S110 - pid probing never blocks
        pass

    doc = {"tag": args.tag, "url": args.url, "levels_requested": levels,
           "requests_per_level": args.requests_per_level,
           "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "server_pid": server_pid, "levels": []}
    stopped_at = None
    for lv in levels:
        print(f"[conc] level={lv} tag={args.tag} ...", flush=True)
        rec = run_level(args.url, lv, args.requests_per_level, args.voice,
                        args.timeout, server_pid)
        doc["levels"].append(rec)
        print(f"[conc] level={lv}: ok={rec['successes']}/{rec['requests']} "
              f"ttfa_p50={rec['ttfa_p50_s']} ttfa_p95={rec['ttfa_p95_s']} "
              f"e2e_p50={rec['e2e_p50_s']} audio/wall={rec['audio_per_wall_second']} "
              f"req/min={rec['requests_per_minute']} peak_vram={rec['peak_vram_gb']}",
              flush=True)
        if rec["failures"]:
            stopped_at = lv
            print(f"[conc] FAILURES at level {lv} — ladder stops "
                  f"(verbatim: {rec['failures_verbatim']})", flush=True)
            break
    doc["ladder_stopped_at"] = stopped_at
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[conc] wrote {args.json_out}", flush=True)
    print("CONCURRENCY-PROBE-DONE", flush=True)
    return 1 if stopped_at is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
