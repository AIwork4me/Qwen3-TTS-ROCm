# vLLM-Omni serving on gfx1151 — measured guide (Radeon 8060S)

**Measured:** 2026-09-24/25 · **Stack:** vllm 0.30.0+rocm723 + vllm-omni
0.30.0rc1 + onnxruntime-rocm (isolated venv; the official qwen-tts `.venv`
is never touched) · **Evidence:**
[serving](../evidence/vllm-online-serving-gfx1151-2026-09-24.txt) ·
[streaming](../evidence/vllm-streaming-gfx1151-2026-09-24.txt) ·
[concurrency](../evidence/vllm-concurrency-gfx1151-2026-09-25.txt) ·
machine records linked from the [evidence index](../evidence/README.md)

This page answers, from measured data only, the serving questions a Radeon
8060S developer actually has. Scope: THIS host (Ryzen AI Max+ PRO 395 iGPU,
unified memory), THIS stack, short-prompt workloads. Nothing here is a
universal-best-configuration claim.

## What works (upstream recipe invocations, unmodified code)

| Path | Status | Key numbers |
|---|---|---|
| `vllm serve …-CustomVoice --omni` + `/v1/audio/speech` / `/v1/audio/voices` | ✅ | ready ~2 min; 10 preset voices; non-streaming WAV 200 OK |
| VoiceDesign serving (natural-language voice description) | ✅ | 200 OK, valid 24 kHz WAV |
| Base inline voice clone (`ref_audio` file:// URI + `ref_text`) | ✅ | needs `--allowed-local-media-path`; raw base64 is rejected by the documented contract |
| HTTP PCM streaming (`stream=true, stream_format=audio, response_format=pcm`) | ✅ TRUE streaming | TTFA 0.249 s short / 0.222 s 453-char input (client-side timestamps) |
| Upstream WebSocket client | ❌ upstream client/server drift | client targets a route this build doesn't expose; `/v1/realtime` errors, `/v1/duplex` closes 1000 — verbatim evidence, not a gfx1151 failure |

## Concurrency on one GPU (8 requests/level, short prompts)

| c | default cfg: TTFA p50 / E2E p50 / req-min | stage-1 `max_num_seqs: 1`: TTFA p50 / E2E p50 / req-min |
|---|---|---|
| 1 | 0.21 s / 3.7 s / 12.3 | 0.21 s / 3.7 s / 12.4 |
| 2 | 0.30 s / 4.9 s / 24.4 | 0.28 s / 4.3 s / 23.3 |
| 4 | 0.44 s / **35.8 s** / 3.8 | 0.47 s / 6.4 s / 28.5 |
| 8 | **15.1 s** / **273.3 s** / 1.8 | **0.74 s / 8.6 s / 34.9** |

**The one tuning that matters on this host:** the deploy config's own
comment ("CustomVoice / VoiceDesign are TTFA-optimal at 1") is decisively
confirmed — setting stage-1 `max_num_seqs: 1` turns the default config's
c≥4 queueing collapse into a stable, monotone ladder that is *better than
realtime in aggregate* at c=8 (1.50 audio-seconds per wall-second). No
failures, OOM, or HIP errors at any level in either configuration.

**Highest stable-and-useful concurrency for this stack:** c=8 with
stage-1 `max_num_seqs: 1` (or c=2 if you must stay on the packaged
default). Both answers are this-host claims.

## Measurement caveats (what these numbers do NOT say)

- Dedicated-VRAM readings (`rocm-smi`) stay ~0.72 GiB flat throughout —
  the engine's memory lives in the unified GTT pool on this APU, so
  dedicated VRAM is not a utilization signal; no memory-pressure numbers
  are claimed. Process-RSS sampling failed (0.0) and is not claimed.
- Short prompts only; long-prompt concurrency was not characterized
  (single-request long-stream cadence is in the Task 9 evidence).
- Latencies are cold-ish single-session medians, not a benchmark-v2-style
  n=5 statistical treatment; treat as serving-profile measurements.
