# Qwen3-TTS-ROCm P0 Capability Parity Report

Program: P0 capability parity, 2026-09-20 — five gated tasks (T0 audit +
Tasks 1–5), subagent-driven, verifier-gated per task.

## North Star

**Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark, with zero upstream patches.**
（CN mirror: **AMD Radeon 上的官方 Qwen3-TTS —— 能力逐项验证、基准逐项实测、零上游补丁。**）

Three commitments make that sentence meaningful. **Official-first**: every
generation flow in this repository goes through the unmodified official
`qwen-tts` package — `loader.load()` returns the native official model
object, and every workflow added by this program composes only the official
inference APIs (`generate_custom_voice`, `generate_voice_design`,
`generate_voice_clone`, `create_voice_clone_prompt`). **Zero-patch**: no
vendored or patched upstream source, ever; the published `qwen-tts==0.1.1`
artifact is depended on as-is, enforced by a dedicated parity test and
verified by pip-RECORD hash audits (below). **Evidence-first**: every
capability claim in the READMEs links a verbatim, machine-generated
transcript under [`evidence/`](../evidence/README.md); what is not proven is
labelled 🟡/⬜/🚫, numbers carry their conditions, and no claim generalizes
beyond the single validated configuration (Radeon 8060S, `gfx1151`, ROCm
7.14.0).

## Repository SHAs

| What | Value |
|---|---|
| Project at program start (Task 0 commit) | `f913d4c` |
| Project before Task 5 | `91dfb36` (Task 4 commit) |
| Project final | **Task 5 commit** (this commit — the alignment commit containing this report) |
| Upstream `QwenLM/Qwen3-TTS`, pinned | `022e286b98fbec7e1e916cb940cdf532cd9f488e` (gitignored clone at `.upstream/Qwen3-TTS`, pristine — `git status --porcelain` empty) |
| Installed `qwen-tts` | `0.1.1` (published artifact, unmodified — RECORD hash audits green at Tasks 1/2/4) |
| Validation host | AMD Ryzen AI Max+ PRO 395 / Radeon 8060S (`gfx1151`), ROCm 7.14.0 (`torch 2.12.0+rocm7.14.0`, HIP 7.14.60850), Python 3.12 |

## Completed

| Task | What it delivered | Verification |
|---|---|---|
| Task 0 — ground-truth audit | Pinned upstream SHA `022e286b…`; upstream `finetuning/` layout recorded (entry points, CLI flags, defaults, checkpoint conversion); runtime `get_supported_languages()` captured; 0.6B CustomVoice `instruct` reality pinned (silently nulled by the installed wrapper at `qwen3_tts_model.py:799-800`); evidence inventory. Commit `f913d4c`. | Read-only audit recorded in [`evidence/ground-truth-2026-09-20.md`](../evidence/ground-truth-2026-09-20.md) |
| Task 1 — 0.6B E2E parity | 9 new on-GPU tests for `custom-voice-0.6b` (metadata, single/batch, kwarg passthrough, pinned `instruct` boundary) and `base-0.6b` (ref-text clone, x-vector-only clone, reusable prompt, save/load roundtrip, batch); 0.6B benchmark (RTF, load seconds, peak allocation). Commit `fb8bc60`. | Verifier: PASS — [`docs/superpowers/reports/task-1-verification.md`](superpowers/reports/task-1-verification.md) |
| Task 2 — multilingual matrix | All 10 officially supported languages exercised end to end on the Radeon GPU (10 CustomVoice + 10 VoiceDesign + 4 representative cross-lingual clone pairs; manifest drift is a hard error, never a silent skip); 13 CPU unit tests. Commit `2a90229`. | Verifier: PASS — [`docs/superpowers/reports/task-2-verification.md`](superpowers/reports/task-2-verification.md) |
| Task 3 — Voice Studio | `voice_workflow` composing only the three official APIs into Design → Clone → Reuse with the official model split; ⑥ Voice Studio demo tab (no download/re-upload); three phases timed separately; +15 CPU / +4 GPU tests. Commit `d5a74b0`. | Verifier: PASS — [`docs/superpowers/reports/task-3-verification.md`](superpowers/reports/task-3-verification.md) |
| Task 4 — fine-tuning smoke | Official `finetuning/` workflow proven to execute on the Radeon GPU: self-generated 12-utterance dataset → `prepare_data.py` as-is → 12 optimizer steps of `sft_12hz.py` (1.7B Base bf16) → checkpoint save → official reload → sane synthesis. Execution validation only; the single disclosed `sdpa` deviation lived inside the gitignored clone and was reverted. Commit `91dfb36`. | Verifier: PASS — [`docs/superpowers/reports/task-4-verification.md`](superpowers/reports/task-4-verification.md) |
| Task 5 — repository alignment | North Star + four-state capability matrix in README/README_CN (every ✅ linked to evidence); CustomVoice/Base conceptual boundary fixed; `patch.py` → `compat.py` rename; stale suite counts recomputed honestly (EN=CN); hardware-humility and roadmap sections; evidence/README errata; verifier reports copied to `docs/superpowers/reports/`; this report. Task 5 commit. | **PASS** — final verification gate ([task-5-verification.md](superpowers/reports/task-5-verification.md)); all 11 criteria independently re-verified incl. full CPU 252/252 + GPU 38/38 re-runs |

## Capability matrix

All ✅ rows were exercised on the validation host (Radeon 8060S, `gfx1151`)
through unmodified official APIs. States: ✅ Radeon E2E validated · 🟡
partial — load-only · ⬜ not validated · 🚫 not exposed upstream or
intentionally not claimed.

| Capability | Model | Radeon status | Evidence |
|---|---|---|---|
| CustomVoice generation (preset / custom speakers) | 1.7B | ✅ E2E validated | [`evidence/gen-customvoice.txt`](../evidence/gen-customvoice.txt) |
| CustomVoice generation | 0.6B | ✅ E2E validated | [`evidence/gpu-suite-2026-09-20.txt`](../evidence/gpu-suite-2026-09-20.txt) |
| VoiceDesign (text-described voice creation) | 1.7B | ✅ E2E validated | [`evidence/gen-voicedesign.txt`](../evidence/gen-voicedesign.txt) |
| Voice Clone — zero-shot cloning from reference audio | 1.7B Base | ✅ E2E validated | [`evidence/gen-voiceclone.txt`](../evidence/gen-voiceclone.txt) |
| Base family (zero-shot cloning + fine-tuning base) | 0.6B | ✅ E2E validated | [`evidence/gpu-suite-2026-09-20.txt`](../evidence/gpu-suite-2026-09-20.txt) |
| Reusable clone prompt (create → save → load → reuse) | 1.7B & 0.6B Base | ✅ E2E validated | [`evidence/gen-voiceclone.txt`](../evidence/gen-voiceclone.txt) · [`evidence/gpu-suite-2026-09-20.txt`](../evidence/gpu-suite-2026-09-20.txt) |
| Design → Clone → Reuse (Voice Studio) | VoiceDesign 1.7B + Base | ✅ E2E validated | [`evidence/voice-workflow-2026-09-20.txt`](../evidence/voice-workflow-2026-09-20.txt) |
| 12Hz tokenizer codec (encode → decode roundtrip) | Tokenizer-12Hz | ✅ E2E validated | [`evidence/tokenizer-codec.txt`](../evidence/tokenizer-codec.txt) |
| Multilingual — all 10 official languages end to end | 1.7B CV + VD + Base | ✅ E2E validated | [`evidence/multilingual-matrix.txt`](../evidence/multilingual-matrix.txt) · [`evidence/multilingual-matrix.json`](../evidence/multilingual-matrix.json) |
| Fine-tuning (official SFT workflow) | 1.7B Base | ✅ scoped — execution-only smoke, no quality claims | [`evidence/finetune-smoke-2026-09-20.txt`](../evidence/finetune-smoke-2026-09-20.txt) · [`evidence/finetune-smoke-2026-09-20.json`](../evidence/finetune-smoke-2026-09-20.json) |
| Benchmark coverage (median RTF, archived runs) | 1.7B trio + 0.6B pair | ✅ archived | [`evidence/benchmark.json`](../evidence/benchmark.json) · [`evidence/benchmark-06b-2026-09-20.json`](../evidence/benchmark-06b-2026-09-20.json) |
| Instruction control on CustomVoice | 0.6B | 🚫 not exposed upstream (wrapper silently nulls `instruct`) | [`evidence/ground-truth-2026-09-20.md`](../evidence/ground-truth-2026-09-20.md) |
| vLLM-Omni serving | — | 🚫 → roadmap (feasibility not started) | README "Roadmap (not yet validated)" |
| True streaming inference | — | 🚫 → roadmap (no Radeon measurements exist) | README "Roadmap (not yet validated)" |

## Benchmarks

Median RTF (wall-clock seconds per second of generated audio; lower is
better; bfloat16/sdpa, `max_new_tokens=512`, short/medium texts, n=2 per
cell — ranges quoted, not point estimates):

| Workload | Model | Median RTF range | Source |
|---|---|---|---|
| Custom Voice | 1.7B | 1.31 – 1.51 | [`evidence/benchmark.json`](../evidence/benchmark.json) (2026-08-27) |
| Voice Design | 1.7B | 1.27 – 1.62 | [`evidence/benchmark.json`](../evidence/benchmark.json) |
| Voice Clone, zero-shot | 1.7B Base | 1.71 – 1.88 | [`evidence/benchmark.json`](../evidence/benchmark.json) |
| Custom Voice | 0.6B | 1.03 – 1.30 (load 4.0 s, peak alloc 2.80 GiB) | [`evidence/benchmark-06b-2026-09-20.json`](../evidence/benchmark-06b-2026-09-20.json) |
| Voice Clone, zero-shot | 0.6B Base | 1.19 – 1.26 (load 1.5 s, peak alloc 3.10 GiB) | [`evidence/benchmark-06b-2026-09-20.json`](../evidence/benchmark-06b-2026-09-20.json) |

Voice-workflow three-phase timings (recorded separately, never collapsed;
[`evidence/voice-workflow-2026-09-20.json`](../evidence/voice-workflow-2026-09-20.json)):
design ≈ 5.5 s · prompt creation ≈ 0.3 s · reuse ≈ 5.3–6.4 s per sentence.

Caveats stated with every number: this is an integrated iGPU on a
shared-memory APU — RTF drifts with clocks, thermals, memory pressure and
background load; the 0.6B run and the 1.7B run come from different sessions
(days apart) and are not a controlled A/B; n=2 per cell is a smoke-scale
sample, not a statistical characterization. Precision is deliberately not
overstated anywhere.

## Newly proven Radeon value

What this program added beyond upstream (which documents CUDA /
FlashAttention deployment):

1. **0.6B end-to-end on a real Radeon** — both 0.6B checkpoints generate,
   clone and persist reusable prompts on gfx1151, with benchmark evidence;
   upstream ships no AMD validation at all.
2. **The full multilingual matrix on a real GPU** — all 10 officially
   supported languages exercised end to end with an archived,
   machine-readable matrix (24/24 cells + 4/4 clone references).
3. **A first-class design → clone → reuse workflow** — the official demo
   never wired its own capabilities end to end; `voice_workflow` + the
   Voice Studio tab do it using only official APIs, with the official model
   split respected.
4. **Official fine-tuning execution proof on ROCm** — the upstream
   `finetuning/` workflow (prep → train → checkpoint → reload → synthesis)
   runs on gfx1151, with the flash-attn blocker documented verbatim and a
   disclosed, reverted, clone-local workaround.
5. **Evidence-first documentation** — a four-state capability matrix where
   every green cell links a verbatim transcript, honest boundary rows for
   what upstream does not expose, and identical EN/CN claims.

## Remaining gaps

* **vLLM-Omni serving on ROCm** — no feasibility work yet; ladder defined
  in the README roadmap (feasibility first, online serving only when
  upstream supports the required path).
* **True streaming** — zero Radeon measurements; upstream's "97 ms" figure
  is not a Radeon number; required measurements listed in the roadmap.
* **Additional Radeon architectures** — everything above is gfx1151-scoped;
  other ROCm hardware is explicitly unvalidated (community reports wanted).
* **Long-run fine-tuning** — only a 12-step execution smoke; no
  convergence, stability or scalability study.
* **Quality evaluation** — no speaker-similarity / MOS / pronunciation
  metrics anywhere; "sane waveform" is a structural check, not a quality
  claim.

## Zero-patch audit

Upstream modifications remain **ZERO**:

* pip RECORD hash audits of the installed `qwen-tts 0.1.1` were re-run
  green by the task verifiers at Tasks 1, 2 and 4 (25/25 and 26/26 source
  files byte-identical, 0 modified).
* The pinned clone at `.upstream/Qwen3-TTS` is pristine at
  `022e286b98fbec7e1e916cb940cdf532cd9f488e` — `git status --porcelain`
  empty, re-verified in the Task 5 final regression.
* The single disclosed temporary deviation (one-line
  `flash_attention_2` → `sdpa` in `finetuning/sft_12hz.py` during the
  Task 4 smoke) lived only inside that gitignored clone, was diff-recorded
  in [`evidence/finetune-smoke-2026-09-20.txt`](../evidence/finetune-smoke-2026-09-20.txt),
  and was reverted before commit (closeout + final verification blocks in
  the transcript; errata for two misleading transcript labels in
  [`evidence/README.md`](../evidence/README.md)).
* The upstream-parity test (`tests/test_official_demo_parity.py`) — stock
  upstream demo built around our loader — is part of the green GPU suite.

## Test summary

| Suite | Count | Result |
|---|---|---|
| CPU (`pytest -m 'not gpu'`) | 252 | all pass on the validation host (includes the 1 HIP-gated test) |
| GPU (`pytest -m gpu`) | 38 | all pass on the validation host (real synthesis, weights required) |
| Validation-host total | 290 | **290 / 290** |
| CPU-only CI (`pytest -m "not gpu and not requires_download"`) | 252 | same 252 tests on Python 3.10/3.11/3.12 with **1 HIP-gated skip** (no AMD GPU on the runner) |
| Lint | — | `ruff check .` clean |
| Subagent verification | 5 tasks | **5 / 5 PASS** — verifier reports committed verbatim at [superpowers/reports/](superpowers/reports/) |

## Changed files

Summary per task commit (details in each commit):

* `f913d4c` (T0): `.gitignore`, `evidence/README.md`,
  `evidence/ground-truth-2026-09-20.md`.
* `fb8bc60` (T1): `scripts/benchmark.py`, `tests/test_benchmark.py`,
  `tests/test_generate_custom_voice_06b.py`, `tests/test_voice_clone_06b.py`,
  `evidence/{benchmark-06b-2026-09-20.json,benchmark-06b-2026-09-20.txt,gpu-suite-2026-09-20.txt}`,
  README/README_CN/CHANGELOG.
* `2a90229` (T2): `scripts/validate_languages.py`,
  `tests/data/multilingual_samples.json`, `tests/test_validate_languages.py`,
  `evidence/multilingual-matrix.{txt,json}`, README/README_CN/CHANGELOG/
  evidence-README.
* `d5a74b0` (T3): `src/qwen3_tts_rocm/voice_workflow.py`,
  `src/qwen3_tts_rocm/demo/{backend.py,ui.py}`,
  `src/qwen3_tts_rocm/testing.py`, `tests/test_voice_workflow.py`,
  `tests/test_demo_backend.py`, `tests/test_demo_ui.py`,
  `evidence/voice-workflow-2026-09-20.{txt,json}`, READMEs + CHANGELOG +
  evidence-README.
* `91dfb36` (T4): `scripts/make_finetune_dataset.py`,
  `tests/data/finetune_manifest.json`, `tests/test_finetune_dataset.py`,
  `docs/finetuning-rocm.md`,
  `evidence/finetune-smoke-2026-09-20.{txt,json}`, READMEs + CHANGELOG +
  evidence-README + `.gitignore`.
* **Task 5 commit** (this commit): README/README_CN (North Star, capability
  matrix, counts, roadmap), `CHANGELOG.md`,
  `src/qwen3_tts_rocm/patch.py` → `src/qwen3_tts_rocm/compat.py` (+
  `loader.py`, `tests/test_loader.py` updated), `scripts/download_models.sh`
  help text, `evidence/README.md` (errata + corrected row),
  `scripts/make_finetune_dataset.py` (lint fixes), new
  `docs/superpowers/reports/task-{1,2,3,4}-verification.md` (verbatim
  copies) and `docs/p0-parity-report.md`.

## Commits

| Commit | Task |
|---|---|
| `f913d4c` | T0 — ground-truth audit (chore/evidence) |
| `fb8bc60` | T1 — 0.6B end-to-end validation + benchmarks |
| `2a90229` | T2 — multilingual capability matrix |
| `d5a74b0` | T3 — Voice Studio (design → clone → reuse) |
| `91dfb36` | T4 — fine-tuning execution smoke |
| Task 5 commit (this commit) | T5 — repository alignment with the North Star + this report |
