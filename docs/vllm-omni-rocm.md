# vLLM-Omni on ROCm — what the controlled A/B showed (gfx1151)

This document answers, from measured data only, the deployment question a
Radeon developer actually faces: **should I run Qwen3-TTS through the
official `qwen_tts` package (this repo's loader) or through vLLM-Omni?**
Every claim below is keyed to rows in
[`evidence/vllm-vs-qwen-tts-2026-09-21.json`](../evidence/vllm-vs-qwen-tts-2026-09-21.json)
(verbatim transcript: `evidence/vllm-vs-qwen-tts-2026-09-21.txt`); nothing
here is a **quality** claim — output quality is explicitly out of scope for
this comparison and is left to the Task 14 metrics work.

## What was compared (method summary)

Same workload, same checkpoint, same warmup policy, one GPU, one continuous
window (JSON `method`, `workload_manifest`, `continuous_run_record`):

| | side A — qwen-tts | side B — vLLM-Omni |
|---|---|---|
| stack | official `qwen_tts` 0.1.1 via `qwen3_tts_rocm.loader.load("custom-voice")` (bf16 + sdpa) | vLLM `0.28.0+rocm723` + vLLM-Omni `0.28.0` (isolated `.work-vllm/venv`), upstream `end2end.py --query-type CustomVoice` invocation pattern, byte-reused |
| checkpoint | `models/Qwen3-TTS-12Hz-1.7B-CustomVoice` | **the same directory** (HF_HOME hub-snapshot symlink) |
| workload | 8 CustomVoice prompts (4 zh / 4 en; 1 short, 2 medium, 1 long each), speaker Vivian, blank instruct | identical manifest |
| cap | `max_new_tokens=512` explicit | `"max_new_tokens": [512]` carried in `additional_information` exactly as `end2end.py` does (field not consumed by the runtime — see fairness note) |
| runs | block A1 + block A2 (3 timed runs per prompt per block, 48 rows) | block B (3 timed runs per prompt, 24 rows) |
| warmup | 1 discarded generation per block (equal policy) | same |
| sampling | checkpoint `generate_config.json` defaults | engine defaults (per-request sampling not settable in the reused pattern) |

Block order **A-B-A** (qwen → vLLM-Omni → qwen) so the qwen side brackets the
vLLM-Omni measurement window on the same heat-soaked GPU; every block ran in
its own process, and the sysfs/`rocm-smi` snapshots at every block boundary
(JSON `block_snapshots`) show GPU memory returned to the 146 MB idle baseline
between blocks — neither side's numbers are contaminated by the other's
residency. Continuous window: 1108 s, all three block exits 0, all 72 runs
completed (`rows`: 48 + 24, zero failures, every waveform finite and
non-silent at 24 kHz).

Host: AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S (`gfx1151`, unified memory,
94 GiB RAM), ROCm 7.2.1, kernel 6.17.0-1032-oem (JSON `environment`).

## 1. What works?

**The full tested path works, reproducibly, on gfx1151.** With the isolated
`.work-vllm/venv` stack installed exactly per the vLLM-Omni ROCm docs
(proven first by `evidence/vllm-omni-feasibility-2026-09-21.json`), the
upstream offline CustomVoice invocation ran **24/24 timed generations to
completion** in block B: every output a finite, non-silent 24 kHz waveform,
RTF range 1.13–1.36 (`side_summary.vllm`, `rows[*].waveform`). Model
resolution from the local checkpoint via the HF_HOME symlink, engine stage
init, talker → Code2Wav pipeline, and clean engine shutdown all worked;
multiple sequential `omni.generate()` calls on one engine instance (25 in a
row) were stable. (JSON: `per_block_summaries.B`, `rows`.)

## 2. What does not (work / hold)?

Within the tested path — recorded as limitations, not verdicts about the
whole project (JSON `unsupported_parity`):

* **No per-request sampling control** in the offline CustomVoice pattern:
  temperature/top-k/top-p cannot be set per request the way the official
  `qwen_tts` API accepts them; the engine defaults apply.
* **The `max_new_tokens` field in `additional_information` is not consumed**
  by the qwen3_tts offline pipeline in vLLM-Omni 0.28.0 (code inspection;
  generation ends on the talker stop token). It is kept for pattern parity,
  but it is not an effective cap knob — see the fairness note below for why
  this A/B is still apples-to-apples.
* **In-process `torch.cuda.max_memory_allocated` sees only the orchestrator
  process** — the workers live in spawned processes, so the meaningful
  whole-stack memory number on this stack is the sysfs counter (identical
  sampler on both sides; JSON `method.memory_measurement`).
* **Not exercised here:** serving (OpenAI/AsyncOmni streaming), VoiceDesign
  and Base (clone) task types, batching (`--batch-size` exists in
  `end2end.py` but per-prompt RTF parity at batch 1 was the question), other
  checkpoints, multi-user concurrency. The upstream deploy config ships
  "Verified on 1x H100" wording — gfx1151 is community territory: it works
  (this evidence), but upstream does not validate it.

## 3. What is faster / slower?

**Steady-state generation: vLLM-Omni is faster on this host — on every
prompt.** Median RTF (wall ÷ audio-seconds; lower is better):

| side | pooled median RTF | per-prompt median range |
|---|---|---|
| qwen-tts (A1+A2, 48 runs) | **1.303** | 1.258 – 1.344 |
| vLLM-Omni (B, 24 runs) | **1.182** | 1.138 – 1.238 |

Per-prompt vLLM/qwen median ratios: 0.846 – 0.986 — vLLM-Omni is 1.4–15%
faster depending on prompt, largest gains on medium/long prompts
(`per_prompt_medians`). The gap survives the thermal drift the A-B-A design
controls for: qwen's own blocks drifted 1.291 → 1.324 across the 18.5-minute
window, and vLLM-Omni still beat qwen's *better* block by ~8%.

**Cold start is the opposite, overwhelmingly:** model load
`loader.load()` **4.8–4.9 s** vs `Omni(...)` stage init **69.1 s** (14×), and
the discarded warmup generation 5.8–6.4 s vs 16.9 s (`side_summary.*.
load_s_by_block`, `warmup_wall_s_by_block`). For a one-shot CLI render,
qwen-tts delivers finished audio before the vLLM-Omni engine has finished
loading in every configuration measured here.

## 4. What consumes more / less memory?

**vLLM-Omni's whole-stack GPU footprint is ~3.8× larger.** Peaks of the
amdgpu `mem_info_vram_used + mem_info_gtt_used` counters (identical 1 Hz
sampler both sides; on this unified-memory APU allocations land almost
entirely in the GTT counter; counter values recorded in MiB in the JSON):

| side | measured-phase peak (vram+gtt) | of which GTT |
|---|---|---|
| qwen-tts (A1 / A2) | **7.0 / 7.1 GiB** | 5.6 / 5.7 GiB |
| vLLM-Omni (B) | **26.3 GiB** | 25.0 GiB |

(`side_summary.*.sysfs_peak_mb_by_block`; ~24.5 GiB of it is preallocated by
the engine at load — the counter is already there before the first measured
generation.) For reference the qwen side's in-process torch peak allocation
is 4.66–4.86 GiB (`peak_alloc_gib_by_block`), consistent with the historical
benchmark evidence. On this 94 GiB-RAM host both fit; a 16–32 GiB-RAM machine
would run the qwen side comfortably and likely **not** fit the vLLM-Omni
side at all.

## 5. When should a Radeon developer use qwen-tts?

* **Short-lived processes / one-shot generation** (CLI, scripts, CI,
  notebooks): 4.8 s load vs 69 s dominates everything else; total
  time-to-audio is lower for any single render.
* **Memory-constrained hosts**: ~7 GiB peak vs ~26 GiB.
* **Anything that needs the official API surface**: keyword-first
  `generate_custom_voice` / `generate_voice_design` / `generate_voice_clone`
  with per-call sampling overrides — parity with the upstream demo and docs.
* **Fine-tuning**: the official `qwen_tts`/`finetuning` stack (this repo's
  Task 4–7 evidence) trains the checkpoint you then load in-process; the
  vLLM-Omni path here is inference-only.
* When you want deterministic-in-process numbers (single process, torch
  counters meaningful).

## 6. When vLLM-Omni?

* **A long-lived service rendering many prompts sequentially**: the 69 s
  load and ~26 GiB residency amortize, and every generation after that is
  ~8–15% faster (RTF 1.18 vs 1.30). Rough break-even: the startup deficit is
  ~75 s (load + warmup difference) and the per-render saving ≈ RTF gap ×
  audio seconds (≈ 0.12 × duration), i.e. ≈0.9 s at the workload's
  pooled-median 7.72 s render (qwen-side median 8.28 s; both from the
  JSON's 72 `rows` — figure corrected from a stale "~8–9 s" eyeball by the
  2026-09-21 claims audit) — the vLLM-Omni engine pays for its startup
  after roughly 50–100 renders.
* **When batching/throughput is the goal**: the offline API accepts batched
  inputs (`end2end.py --batch-size/--use-batch-sample`) where batching
  benefits are plausible — but this A/B did not measure batching (see
  unproven below), so size it yourself before committing.
* When the vLLM ecosystem around it (continuous batching, an OpenAI-compatible
  serving path in-tree) is the operational fit — noting the serving path is
  itself untested on this host.

## 7. Which conclusions are still unproven?

Deliberately out of scope or unmeasured here — do **not** quote this
document for any of these:

* **Output quality / speaker similarity / pronunciation** of either side —
  no quality metric was computed; Task 14 builds those. (Both sides produced
  finite, non-silent speech-duration waveforms; that is a sanity check, not
  a quality result.)
* **Batch / multi-request concurrency throughput** on either stack —
  supported by both APIs, not exercised (`method.batch_concurrency`).
* **Streaming / time-to-first-audio**: the vLLM-Omni side has an AsyncOmni
  streaming mode (`end2end.py --streaming`) that this offline A/B did not
  touch; on the qwen side streaming is not exposed by the official API
  (`evidence/streaming-2026-09-21.json`).
* **Serving (OpenAI-compatible) on ROCm**, VoiceDesign / Base (clone) task
  types, the 0.6B checkpoints, discrete-Radeon (dedicated-VRAM) behavior,
  and long-duration stability of the vLLM-Omni engine (25 sequential
  generations is a smoke, not a soak).
* **Sampling-equivalence of outputs**: the two stacks sample with their own
  defaults (fairness note), so waveform contents differ; this A/B compares
  timing and memory only.

## Fairness notes (read before quoting deltas)

From JSON `fairness_notes`:

1. `max_new_tokens=512` is enforced on the qwen side; on the vLLM-Omni side
   the field is carried (pattern parity) but not consumed — effective
   token-cap equivalence does **not** hold by knob. It holds **in practice**
   for this workload: the longest output on either side was 33.5 s ≈ 402
   tokens at 12 Hz < 512, so **neither side was truncated by any cap** and
   every run ended naturally at EOS.
2. Sampling defaults differ per stack (checkpoint `generate_config.json` vs
   engine defaults); audio durations for the same text differ accordingly
   (e.g. p4: ~30 s vs ~30–34 s). RTF normalizes by produced audio, so the
   timing comparison is unaffected, but the waveforms are not comparable
   samples.
3. Same checkpoint directory both sides; blank instruct both sides (blank =
   absent in both stacks); same speaker (Vivian), same language field.
4. vLLM-Omni wall time covers its full per-request pipeline (prefill +
   decode + Code2Wav), mirroring the qwen side's single end-to-end API call.
5. Alias note: the registry alias for this checkpoint is `custom-voice`
   (there is no `custom-voice-1.7b` alias — `custom-voice` *is* the 1.7B
   CustomVoice checkpoint).
