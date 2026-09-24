#!/usr/bin/env python3
"""vLLM-Omni streaming probe for gfx1151 (v0.2.1 Task 9D/9F).

Drives the UPSTREAM-documented HTTP streaming interface exactly as the
recipe's curl does — ``POST /v1/audio/speech`` with ``stream=true,
stream_format="audio", response_format="pcm"`` — but from Python with
``requests`` iter_content so EVERY response chunk carries a client-side
monotonic arrival timestamp. Streaming is measured from the CLIENT side,
never inferred from server logs.

Measured per request: request start, first response byte, first complete
audio chunk, TTFA (first byte of audio payload), chunk count, per-chunk byte
sizes, inter-chunk gap distribution (median/max/p95), total response wall,
decoded audio duration (PCM s16le mono: bytes / 48000 s at 24 kHz), total
RTF (wall/audio), longest gap, and a playback-buffer simulation
(real-time consumption at 48 000 B/s starting at first audio byte) counting
simulated underruns (buffer empties before the stream ends) and minimum
buffer depth.

Non-streaming sanity companion mode (--mode nonstream) records a single
POST's wall time + WAV/PCM size for TTFA-vs-total contrast: if TTFA ≈ total
wall, there is NO incremental streaming.

Usage::

    python scripts/vllm_streaming_probe.py --url http://127.0.0.1:8091 \
        --input "Hello, how are you?" --voice vivian --language English \
        --json-out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import requests

BYTES_PER_SECOND_PCM = 24_000 * 2  # s16le mono


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="vLLM-Omni streaming probe (Task 9)")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--mode", choices=("stream", "nonstream"), default="stream")
    ap.add_argument("--input", required=True)
    ap.add_argument("--voice", default="vivian")
    ap.add_argument("--language", default="English")
    ap.add_argument("--instructions", default=None)
    ap.add_argument("--task-type", default="CustomVoice",
                    help="CustomVoice | VoiceDesign | Base (matches served model)")
    ap.add_argument("--ref-audio-path", default=None, help="Base inline clone: wav path")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="")
    return ap.parse_args(argv)


def build_body(args) -> dict:
    body: dict = {"input": args.input, "task_type": args.task_type}
    if args.task_type == "CustomVoice":
        body["voice"] = args.voice
        if args.language:
            body["language"] = args.language
        if args.instructions:
            body["instructions"] = args.instructions
    elif args.task_type == "VoiceDesign":
        body["instructions"] = args.instructions or "A warm, friendly female voice"
        if args.language:
            body["language"] = args.language
    elif args.task_type == "Base":
        if args.ref_audio_path:
            # Upstream API contract (learned live in Task 9C, 400 verbatim):
            # ref_audio must be a URL, data:...base64 URL, or file:// URI —
            # raw base64 is rejected. file:// is the cleanest local form.
            body["ref_audio"] = "file://" + str(args.ref_audio_path)
            if args.ref_text:
                body["ref_text"] = args.ref_text
    if args.max_new_tokens:
        body["max_new_tokens"] = args.max_new_tokens
    return body


def probe_stream(args) -> dict:
    body = build_body(args) | {"stream": True, "stream_format": "audio",
                               "response_format": "pcm"}
    t0 = time.monotonic()
    chunks: list[tuple[float, int]] = []  # (arrival monotonic, bytes)
    first_byte = None
    total = 0
    with requests.post(f"{args.url}/v1/audio/speech", json=body,
                       stream=True, timeout=args.timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            now = time.monotonic()
            if first_byte is None:
                first_byte = now
            chunks.append((now, len(chunk)))
            total += len(chunk)
    t_end = time.monotonic()

    arrivals = [c[0] for c in chunks]
    sizes = [c[1] for c in chunks]
    gaps = [round(arrivals[i] - arrivals[i - 1], 4) for i in range(1, len(arrivals))]
    audio_s = total / BYTES_PER_SECOND_PCM
    wall = t_end - t0
    ttfa = (first_byte - t0) if first_byte else None

    # Playback-buffer simulation: consume 48 000 B/s from first audio byte.
    # min_buffer_bytes tracks the shallowest buffer level BEFORE each refill
    # (0.0 == never dipped below the previous chunk's credit; archived v0.2.1
    # JSONs recorded a by-construction 0.0 — kept unedited).
    underruns = 0
    min_buffer = None
    if chunks:
        buf = 0.0
        t_play = first_byte
        for arrival, size in chunks:
            buf -= (arrival - t_play) * BYTES_PER_SECOND_PCM
            if buf < 0:
                underruns += 1 if buf < -4096 else 0  # ignore <1-chunk dips
                buf = 0.0
            min_buffer = buf if min_buffer is None else min(min_buffer, buf)
            buf += size
            t_play = arrival
        min_buffer = 0.0 if min_buffer is None else round(min_buffer, 1)

    rec = {
        "tag": args.tag, "mode": "stream", "input_chars": len(args.input),
        "task_type": args.task_type,
        "ttfa_seconds": round(ttfa, 3) if ttfa else None,
        "first_chunk_seconds": round(chunks[0][0] - t0, 3) if chunks else None,
        "chunk_count": len(chunks),
        "chunk_bytes_total": total,
        "chunk_size_min": min(sizes) if sizes else 0,
        "chunk_size_max": max(sizes) if sizes else 0,
        "chunk_size_median": statistics.median(sizes) if sizes else 0,
        "interchunk_gap_count": len(gaps),
        "interchunk_gap_median_s": round(statistics.median(gaps), 4) if gaps else None,
        "interchunk_gap_max_s": round(max(gaps), 4) if gaps else None,
        "interchunk_gap_p95_s": round(statistics.quantiles(gaps, n=20, method="inclusive")[18], 4)
        if len(gaps) > 1 else None,
        "total_wall_seconds": round(wall, 3),
        "decoded_audio_seconds": round(audio_s, 3),
        "total_rtf_wall_over_audio": round(wall / audio_s, 3) if audio_s else None,
        "ttfa_over_wall": round(ttfa / wall, 3) if ttfa and wall else None,
        "playback_sim": {"underruns": underruns,
                         "min_buffer_bytes": round(min_buffer, 1)},
    }
    return rec


def probe_nonstream(args) -> dict:
    body = build_body(args)
    t0 = time.monotonic()
    r = requests.post(f"{args.url}/v1/audio/speech", json=body, timeout=args.timeout)
    wall = time.monotonic() - t0
    r.raise_for_status()
    return {"tag": args.tag, "mode": "nonstream", "input_chars": len(args.input),
            "task_type": args.task_type,
            "total_wall_seconds": round(wall, 3),
            "response_bytes": len(r.content),
            "content_type": r.headers.get("content-type", "")}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rec = probe_stream(args) if args.mode == "stream" else probe_nonstream(args)
    rec["when_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(json.dumps(rec, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    # Streaming verdict comes from DATA: ttfa << total_wall means incremental.
    if args.mode == "stream" and rec["ttfa_seconds"] is not None and rec["total_wall_seconds"]:
        frac = rec["ttfa_over_wall"]
        verdict = "INCREMENTAL" if (frac or 1) < 0.8 else "NOT-INCREMENTAL"
        print(f"STREAM-VERDICT: {verdict} (ttfa/wall={frac})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
