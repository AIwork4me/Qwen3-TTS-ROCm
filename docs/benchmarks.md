# Benchmarks — RTF on Radeon 8060S (gfx1151)

Real-time factor (RTF) = wall-clock seconds per second of generated audio.
RTF > 1.0 means synthesis is slower than realtime; lower is better. Every
number below was produced live by [`scripts/benchmark.py`](../scripts/benchmark.py)
on the machine described in [Platform](#platform) (recorded session
**2026-08-27T17:12:40+08:00**) and archived verbatim in the evidence files
listed under [Artifacts](#artifacts). **Numbers drift with clock, thermals,
memory pressure and background load on an integrated GPU — treat any single
session as indicative, and cite the artifact + date alongside any number you
quote.**

Headline: for short-to-medium sentences (≤ ~18 s of audio) the tuned-voice
models render at roughly **1.3–1.6× realtime**, i.e. a few seconds of wait for
a few seconds of speech. The zero-shot Base (voice-clone) model lands around
**1.7–1.9× realtime** on the same texts.

## Platform

Pinned facts as reported by the environment during the recorded session —
the raw values are reproduced in `evidence/benchmark.json → meta`.

| Fact | Value |
|---|---|
| CPU | AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S (`/proc/cpuinfo`) |
| APU class | Strix Halo, **unified memory**: CPU and iGPU share one LPDDR5X pool |
| Host RAM | 94.1 GiB `MemTotal` (torch HIP-allocatable pool: 80.0 GiB) |
| GPU | AMD Radeon 8060S Graphics, arch `gfx1151`; torch reports `multi_processor_count=20` (marketing count is 40 CUs; HIP exposes the value above as-is — recorded unmodified) |
| Kernel | Linux 6.17.0-1032-oem x86_64 |
| Torch | `2.12.0+rocm7.14.0` (AMD ROCm wheel index), `torch.version.hip = 7.14.60850` |
| Precision / attention | `bfloat16` weights + `sdpa` (loader smart defaults); **no flash-attn on ROCm** → manual PyTorch attention path; experimental AOTriton sdpa paths left OFF this session |
| Python | 3.12 (`.venv` in-repo) |

## Methodology

* One model load per alias via `qwen3_tts_rocm.loader.load` (official
  `from_pretrained`, bf16/sdpa defaults); teardown between aliases
  (`loader.unload`). In-session loads took 2.2–4.9 s each.
* **Warmup policy:** one cn/short generation per alias, executed then
  *discarded* (kernel/code-path warmup), before any measurement. Warmups
  cost 6.1 s (custom-voice), 4.4 s (voice-design) — and **1396 s for base**
  (a degenerate sampling loop under the official default 2048-token budget:
  the warmup deliberately omits `max_new_tokens`; see below).
* **What counts:** `wall` is the full official call
  (`generate_custom_voice` / `generate_voice_design` / `generate_voice_clone`)
  — text encoding, decode, codec-to-waveform — via `time.perf_counter`.
  `audio` is `len(wav)/sr` of the returned waveform. Per-call first-token
  latency is NOT measurable through the non-streaming official API and is
  deliberately not reported (see [Interpretation](#interpretation)).
* **Sampling kwargs (official-neutral worst case):** `do_sample=True`,
  `temperature=1.0` passed explicitly; `top_k/top_p/repetition_penalty` and
  sub-talker knobs stay with the models' own `generate_config.json` defaults.
  Nothing is tuned for speed or brevity.
* **Token cap:** `max_new_tokens=512` (project latency guardrail). At 12 Hz
  this caps any render near ~42 s of audio while bounding degenerate loops:
  Task 13 measured a single 2048-token runaway at ~23 min; this session's
  discarded base warmup reproduced it almost exactly (1396 s at temperature
  1.0). All *measured* runs in the tables completed normally.
* Each cell (alias × language × length) ran **n=2** measured generations;
  reported are the per-run RTFs, their median (headline), and best/worst
  spread. Texts: fixed inline CN/EN samples (~20 chars "short", ~60 chars
  "medium"; exact strings in `evidence/benchmark.json → meta.args.texts`).

## Results (2026-08-27 session)

Rows below are exactly what `scripts/benchmark.py` printed (wall medians in
seconds; audio durations of these cells ranged 1.28–40.88 s):

| alias | lang | len | runs | median RTF | best RTF | worst RTF | median wall s |
|---|---|---|---|---|---|---|---|
| custom-voice | cn | short | 2 | 1.37 | 1.28 | 1.46 | 3.79 |
| custom-voice | cn | medium | 2 | 1.38 | 1.37 | 1.40 | 18.50 |
| custom-voice | en | short | 2 | 1.51 | 1.28 | 1.74 | 1.94 |
| custom-voice | en | medium | 2 | 1.31 | 1.29 | 1.32 | 4.60 |
| voice-design | cn | short | 2 | 1.27 | 1.26 | 1.29 | 4.64 |
| voice-design | cn | medium | 2 | 1.40 | 1.40 | 1.40 | 23.11 |
| voice-design | en | short | 2 | 1.62 | 1.54 | 1.69 | 2.90 |
| voice-design | en | medium | 2 | 1.38 | 1.31 | 1.45 | 5.37 |
| base | cn | short | 2 | 1.73 | 1.69 | 1.77 | 15.16 |
| base | cn | medium | 2 | 1.71 | 1.68 | 1.75 | 48.61 |
| base | en | short | 2 | 1.88 | 1.86 | 1.90 | 9.91 |
| base | en | medium | 2 | 1.81 | 1.80 | 1.81 | 15.10 |

Per-alias reading:

* **custom-voice** (`generate_custom_voice`, 1.7B): median RTF 1.31–1.51,
  best-behaved profile; ~4 s wall for a short sentence, ~19 s for ~13 s of CN
  speech.
* **voice-design** (`generate_voice_design`, 1.7B): median RTF 1.27–1.62,
  statistically indistinguishable from custom-voice despite the extra style
  instruction.
* **base** (`generate_voice_clone`, ICL-mode clone from the bundled synthetic
  reference clip): consistently higher, median RTF 1.71–1.88, and pays
  reference encoding + longer prompt context every call. Also carries the
  degenerate-loop risk demonstrated by its discarded warmup (1396 s).

## Interpretation

* **iGPU expectations.** RTF ≈ 1.3–1.9 means renders take somewhat longer
  than the speech they produce: seconds of waiting for short replies (tables
  above), tens of seconds for paragraphs. That is usable for interactive
  demos at sentence scale and fully fine for offline/batch synthesis; it is
  not a "many-times-realtime" experience like a discrete dGPU would give.
  The bottleneck is memory-bandwidth-class compute over shared LPDDR5X with
  no flash-attn path available (manual PyTorch attention on ROCm).
* **Honest scope note.** These numbers say nothing about *first-token* /
  perceived latency: the official API used here is non-streaming, so only
  total wall time is observable. Anything interactive adds codec/streaming
  considerations outside this benchmark.
* **Variance disclosure.** n=2 per cell; en/short cells have the smallest
  audio (1.28–2.0 s) so a fraction-of-a-second jitter moves their RTF the
  most (spread up to ±0.23). Desktop compositing, browsers and thermal state
  share the same memory pool as the iGPU. Rerun on an idle machine before
  drawing fine conclusions; quote artifact + date when citing numbers.
* **Comparison guidance vs dGPU.** Do not read these rows as "Qwen3-TTS is
  slow": dedicated ROCm/CUDA GPUs with local VRAM and flash-attn typically
  land well below RTF 1.0 for a 1.7B TTS decoder. The meaningful comparison
  for gfx1151 owners is across *their own* config changes (dtype, attention
  impl, token cap, driver/toolchain updates) using the reproduce block below.

## Reproduce

```bash
# default session: all three aliases, max_new_tokens=512, warmup on, 2 runs/cell
.venv/bin/python scripts/benchmark.py \
  |& tee evidence/benchmark-run.txt     # JSON lands in evidence/benchmark.json

# faster focused run
.venv/bin/python scripts/benchmark.py --aliases custom-voice --runs 3 \
  --max-new-tokens 256 --json-out evidence/my-bench.json
```

Options: `--aliases` (comma list; `custom-voice,voice-design,base`),
`--max-new-tokens` (default 512 guardrail), `--runs` (default 2),
`--warmup/--no-warmup` (default on), `--json-out`
(default `evidence/benchmark.json`). The script prints markdown-ready table
rows so a rerun pastes straight into this file.

## Artifacts

| Artifact | Contents |
|---|---|
| `evidence/benchmark-run.txt` | Raw stdout+stderr of the recorded session: per-call `[bench]` lines, markdown table, MIOpen/ck noise, exit code |
| `evidence/benchmark.json` | Machine-readable `{meta:{host,gpu,cpu,torch_version_hip,date,args,...}, results:[per-cell RTF lists + summaries]}` |
| `scripts/benchmark.py` | Generator for both of the above |

Related evidence: this page's Task-13 figure of a ~23 min (1392 s) single-call
runaway comes from that task's session record
(`.superpowers/sdd/2026-08-27-qwen3-tts-rocm/task-13-report.md`, which used the
official 2048-token default). This benchmark session's own equivalent is the
raw line `[bench] warmup alias=base took=1396.0s (discarded)` in
`evidence/benchmark-run.txt` — same uncapped default budget during warmup.
(`evidence/gen-voiceclone.txt` holds only the post-fix, capped Task-13 rerun.)
