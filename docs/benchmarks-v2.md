# Benchmarks v2 — controlled reproducibility (gfx1151)

**Run date:** 2026-09-24 · **Script:** [`scripts/benchmark_v2.py`](../scripts/benchmark_v2.py) ·
**Evidence:** [`evidence/benchmark-v2-gfx1151-2026-09-24.txt`](../evidence/benchmark-v2-gfx1151-2026-09-24.txt) ·
machine-readable: [`evidence/benchmark-v2-gfx1151-2026-09-24.json`](../evidence/benchmark-v2-gfx1151-2026-09-24.json)

Benchmarks v1 ([`docs/benchmarks.md`](benchmarks.md), [`scripts/benchmark.py`](../scripts/benchmark.py))
established the RTF baseline with n=2 runs per cell and honest warnings that
single sessions drift with thermals and background load. v2 keeps everything
that worked in v1 and adds the controls needed to talk about *variance*, not
just point estimates. v1 is untouched and remains what the `full-weekly` CI
replication step runs — v2 is the deeper, manual-cadence methodology.

## Methodology (binding decisions)

- **Fixed prompt manifest** — the same v1 `TEXTS` grid (Chinese/English ×
  short/medium), so v1 and v2 cells are directly comparable.
- **Coverage: all five registered TTS aliases** — 1.7B and 0.6B per family
  (CustomVoice, VoiceDesign, Base). Results are per-checkpoint; there is
  deliberately **no single "Radeon performance" number**.
- **Fixed seeds per run** — `torch.manual_seed(seed_base + k)` before every
  generation (the official API's only seed surface is the global torch RNG).
  The seed list is identical every session, so the sampled token streams are
  reproducible; seeds differ *across* the 5 runs of a cell so the spread
  reflects real generation variance, not just scheduler noise.
- **One explicit warmup per alias** — the first post-load generation
  (cn/short), timed and **recorded in the JSON** but excluded from the
  measured statistics. Unlike v1's deliberately uncapped warmup (official
  2048-token default; a degenerate base run can burn ~23 min), v2's warmup
  keeps the 512-token latency guardrail.
- **n=5 measured runs per cell** (v1 default was 2), official-neutral
  sampling: `do_sample=True, temperature=1.0`, `max_new_tokens=512`.
- **Phase separation** — cold-start **load time**, **warmup time**, and
  **measured warm-model generation time** are three distinct recorded
  numbers; nothing is silently discarded.
- **Environment bookends** — `rocm-smi` VRAM + temperature/clock snapshots
  before load and after the last generation of every alias, kept verbatim in
  the JSON (sensor availability varies by driver; nothing invented), plus
  per-alias torch peak allocated/reserved.

## Results (RTF = wall seconds / audio seconds produced; lower is better)

| alias | lang | len | n | RTF median | min | max | mean | stdev | p10 | p90 | load s | warmup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| custom-voice | cn | short | 5 | 1.25 | 1.21 | 1.26 | 1.24 | 0.02 | 1.22 | 1.26 | 4.62 | 5.43 |
| custom-voice | cn | medium | 5 | 1.25 | 1.23 | 1.27 | 1.25 | 0.02 | 1.23 | 1.27 | 4.62 | 5.43 |
| custom-voice | en | short | 5 | 1.27 | 1.24 | 1.38 | 1.29 | 0.05 | 1.25 | 1.34 | 4.62 | 5.43 |
| custom-voice | en | medium | 5 | 1.2 | 1.19 | 1.26 | 1.22 | 0.03 | 1.19 | 1.25 | 4.62 | 5.43 |
| custom-voice-0.6b | cn | short | 5 | 1.04 | 1.03 | 1.09 | 1.06 | 0.03 | 1.03 | 1.09 | 2.29 | 3.54 |
| custom-voice-0.6b | cn | medium | 5 | 1.06 | 1.0 | 1.14 | 1.08 | 0.06 | 1.02 | 1.14 | 2.29 | 3.54 |
| custom-voice-0.6b | en | short | 5 | 1.1 | 1.05 | 1.27 | 1.13 | 0.08 | 1.07 | 1.21 | 2.29 | 3.54 |
| custom-voice-0.6b | en | medium | 5 | 1.04 | 1.01 | 1.07 | 1.04 | 0.02 | 1.02 | 1.06 | 2.29 | 3.54 |
| voice-design | cn | short | 5 | 1.26 | 1.21 | 1.36 | 1.28 | 0.07 | 1.22 | 1.36 | 2.17 | 4.47 |
| voice-design | cn | medium | 5 | 1.24 | 1.23 | 1.33 | 1.27 | 0.05 | 1.23 | 1.33 | 2.17 | 4.47 |
| voice-design | en | short | 5 | 1.28 | 1.24 | 1.32 | 1.28 | 0.04 | 1.24 | 1.32 | 2.17 | 4.47 |
| voice-design | en | medium | 5 | 1.19 | 1.17 | 1.26 | 1.21 | 0.04 | 1.18 | 1.24 | 2.17 | 4.47 |
| base | cn | short | 5 | 1.29 | 1.26 | 1.3 | 1.28 | 0.02 | 1.26 | 1.3 | 2.19 | 9.75 |
| base | cn | medium | 5 | 1.3 | 1.26 | 1.4 | 1.33 | 0.06 | 1.27 | 1.4 | 2.19 | 9.75 |
| base | en | short | 5 | 1.52 | 1.33 | 1.58 | 1.46 | 0.11 | 1.34 | 1.56 | 2.19 | 9.75 |
| base | en | medium | 5 | 1.35 | 1.29 | 1.48 | 1.38 | 0.09 | 1.3 | 1.48 | 2.19 | 9.75 |
| base-0.6b | cn | short | 5 | 1.13 | 1.1 | 1.17 | 1.13 | 0.03 | 1.1 | 1.16 | 1.75 | 8.26 |
| base-0.6b | cn | medium | 5 | 1.26 | 1.14 | 1.27 | 1.23 | 0.05 | 1.18 | 1.27 | 1.75 | 8.26 |
| base-0.6b | en | short | 5 | 1.19 | 1.13 | 1.37 | 1.21 | 0.1 | 1.13 | 1.3 | 1.75 | 8.26 |
| base-0.6b | en | medium | 5 | 1.18 | 1.15 | 1.28 | 1.19 | 0.05 | 1.15 | 1.25 | 1.75 | 8.26 |

Per-alias torch peaks (whole measured block, warm model): custom-voice
4.46 GiB allocated / 4.58 reserved · custom-voice-0.6b 2.85/3.11 GiB ·
voice-design 4.78/5.05 GiB · base 5.12/5.41 GiB · base-0.6b 3.27/3.56 GiB
(exact values in the JSON; `rocm-smi` bookends verbatim alongside).

## What these numbers measure — and what they do NOT

**Measured:** wall-clock time of one complete official-API generation call
per run (text encoding + decode + codec-to-waveform), the audio duration
that call produced, and their ratio (RTF), as a function of checkpoint,
language, and prompt length, on *this host* (Radeon 8060S iGPU, bf16 +
SDPA, torch 2.12.0+rocm7.14.0, quiet desktop). Cold-start load time and
first-generation warmup time are reported separately so warm numbers are
never passed off as cold or vice versa.

**NOT measured:** first-token/TTFA latency (the official Python API is
non-streaming — nothing is emitted until completion); serving concurrency
(see the vLLM-Omni work for that); energy; quality (RTF says nothing about
how the audio sounds); performance on any other GPU, driver, or host —
**no extrapolation beyond gfx1151 is implied by any number here**.

**Statistical caveat (read before quoting p10/p90):** each cell is n=5.
Median/min/max/mean/stdev are ordinary order statistics and summaries of
those five observations. p10/p90 are *inclusive linear interpolations of a
five-point sample* — rough order-statistic hints consistent with min/max,
**not** confident tail estimates; a p95 or "worst-case latency" claim
cannot be supported by n=5 and is deliberately not made. Clone-family
(Base) cells additionally mix model-decided render lengths (a stochastic
long render inflates that run's RTF); per-run audio durations are in the
JSON for exact auditing.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_v2.py --json-out /tmp/v2.json
```

The verifier for v0.2.1 Task 4 recomputed every table row above directly
from the JSON runs arrays (see
[`docs/superpowers/reports/pc021-task-4-verification.md`](superpowers/reports/pc021-task-4-verification.md)).
