# P0 Capability Parity Program — Design

- **Date:** 2026-09-20
- **Status:** Approved in brainstorming (all sections); pending spec review
- **Scope:** Full 5-task gated program, executed to completion in one run
- **Repo:** `AIwork4me/Qwen3-TTS-ROCm` (local: `/home/amd/Desktop/Qwen3-TTS-ROCm`, branch `main`)
- **Upstream:** `QwenLM/Qwen3-TTS` (never modified)

## North Star

> **Official Qwen3-TTS on AMD Radeon — capability by capability, benchmark by benchmark,
> with zero upstream patches.**

The user ultimately runs the official Qwen3-TTS models, official `qwen-tts` package,
official public APIs, and official model weights. This repo adds Radeon/ROCm enablement,
validation, reproducibility, UX and benchmarks — NOT a forked AMD implementation.

## Ground truth established at design time

| Fact | Value |
|---|---|
| Project HEAD | `bbcebfe` (clean, synced with `origin/main`) |
| Upstream HEAD | fetched from `QwenLM/Qwen3-TTS` at execution time (Task 0 records it) |
| qwen-tts | pinned `0.1.1` in `pyproject.toml`, installed in repo `.venv` |
| torch / ROCm | `2.12.0+rocm7.14.0` on ROCm 7.14.0 |
| GPU | AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S, `gfx1151` (unified memory) |
| Checkpoints | all 6 already under `models/` (incl. both 0.6B variants + tokenizer) |
| Current test state | 238/238 on validation host (213 CPU + 25 real-GPU) |
| Upstream languages | 10 — Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| 0.6B capability boundary | 0.6B CustomVoice has **no instruction control** upstream (1.7B does) |
| Upstream finetuning | `finetuning/` directory exists at upstream root |
| vLLM-Omni upstream | offline inference only — out of scope this run |
| Streaming upstream | supported upstream with a "97 ms" claim — never reused as a Radeon number |

## Resolved decisions

1. **Session scope:** full mission run — Tasks 0–5 to completion, autonomously until all
   verification gates PASS or a task hits a genuine hardware/evidence boundary.
2. **Git policy:** one logical commit per independently verified task, local on `main`;
   **single push at the end** after Task 5 PASS and the P0 report exist. Nothing leaves
   the machine before that.
3. **Execution architecture (Approach B — subagent-driven):** per task, the orchestrator
   dispatches a fresh implementation subagent and then a fresh independent verification
   subagent. The orchestrator never verifies its own program's tasks; commits are made by
   the orchestrator only after verifier PASS.
4. **Working directory:** all work in `/home/amd/Desktop/Qwen3-TTS-ROCm`. The session's
   original cwd is an unrelated sibling project and stays untouched.

## Process architecture

### Non-negotiable execution rule

Tasks run strictly sequentially. Task N+1 never starts until Task N's verification
subagent returns PASS. Forbidden: implementing ahead of verification, "mostly works",
"the code looks right" without execution evidence, marking capabilities validated
because an upstream API exists, weakening/deleting failing tests without documented
technical reason, README claims before evidence exists, fabricating GPU results.

If hardware required for a step is unavailable, that step stops at the honest boundary:

```
BLOCKED: hardware evidence unavailable
```

Missing validation is never converted into a green claim.

### Per-task state machine

1. Orchestrator writes a **self-contained implementation brief** (spec excerpt +
   acceptance criteria + environment facts + accumulated field notes — subagents start
   cold, so briefs carry everything).
2. **Implementation subagent** (general-purpose, full tools): implements with TDD where
   the repo's test culture expects it, runs CPU tests, runs the task's real-GPU tests,
   generates evidence files, updates docs. **Does not commit.**
3. **Verification subagent** (fresh, independent) receives the brief's verifier contract
   verbatim: assume the implementing agent may be wrong; review the git diff, acceptance
   criteria, tests, raw evidence, documentation claims; try to falsify; do not modify
   code; re-run tests; return PASS or FAIL only after checking every criterion, in the
   mandatory report format (verdict, acceptance criteria checklist, code reviewed, tests
   executed with commands + exit codes, runtime evidence reviewed, claims audit,
   regression audit, problems found, final justification). A PASS without test/evidence
   details is invalid.
4. On FAIL: a fix round carries the verifier's findings back to the implementer
   (resumed via message when its context helps, fresh subagent otherwise), then a **new**
   verifier — repeat until PASS.
5. On PASS: orchestrator commits (one logical commit per task), updates field notes,
   proceeds.
6. After Task 5 PASS + P0 report: **single push**.

### Regression gate (every task)

Verifier re-runs: the full CPU suite, the zero-upstream-patch parity test explicitly,
the task's own new GPU tests, and prior tasks' GPU tests relevant to touched paths.
The **full GPU suite** (original 25 + everything added by Tasks 1–4) re-runs at the
Task 5 final gate.

### Evidence policy

Every new GPU capability produces `evidence/<task>-<date>.json` and/or `.txt`
containing: date; git HEAD at run time (noting the task commit follows); upstream
SHA/version; qwen-tts version; Python; torch; ROCm/HIP; GPU; gfx target; exact
command; exit code; relevant runtime results. Machine-readable JSON preferred alongside
raw terminal transcripts. Transcripts are never hand-edited; documentation may
summarize them.

### Claim vocabulary (repository-wide)

- ✅ Radeon E2E validated — real GPU, real generation, archived evidence
- 🟡 Structurally supported / load-tested / partial validation
- ⬜ Not validated
- 🚫 Not currently exposed upstream / intentionally not claimed

"Supported" is never conflated with "validated". Pronunciation quality is never claimed
from waveform sanity — the multilingual claim is exactly "end-to-end generation
completed on Radeon".

### Scope guard (out of scope this run)

vLLM-Omni on ROCm and true streaming get clearly-labeled roadmap entries at Task 5
only. The upstream "97 ms" streaming figure is never presented as a Radeon number.

## Task 0 — Ground-truth audit (prerequisite, not a gated task)

Read-only audit subagent inspects the repo and current upstream before any
modification, recording: current project HEAD; upstream HEAD SHA; qwen-tts version
actually used; upstream's current model repositories, API surface, CustomVoice /
VoiceDesign / Base behavior, `create_voice_clone_prompt`, supported languages, 0.6B vs
1.7B differences, tokenizer API, fine-tuning workflow layout, streaming limitations,
vLLM-Omni status; inventory of existing validated hardware evidence.
Output: `evidence/ground-truth-2026-09-20.md`, committed as its own
`chore(evidence)` commit. Old notes are not assumed correct.

## Task 1 — Complete 0.6B end-to-end parity

**Goal:** turn 0.6B from "load validated" into real Radeon E2E capability validation.

### 1A. CustomVoice 0.6B — new GPU tests (`tests/test_generate_custom_voice_06b.py`)

- loader returns the native official object;
- `get_supported_speakers()` / `get_supported_languages()` behave (non-empty; consistent
  with upstream semantics);
- single synthesis: waveform finite, non-silent, valid sample rate, bounded duration;
- batch synthesis (upstream accepts lists — validated as actually supported);
- public generation-kwargs passthrough;
- **`instruct` reality test:** upstream documents no instruction control on 0.6B
  CustomVoice. The test pins whatever the installed package actually does (raise vs
  ignore) and documentation/UI communicate that boundary. Parity is not pretended where
  upstream itself does not provide it.

### 1B. Base 0.6B / Voice Clone — new GPU tests (`tests/test_voice_clone_06b.py`)

- `ref_audio` + `ref_text` cloning;
- `x_vector_only_mode=True`;
- `create_voice_clone_prompt` on 0.6B and reuse across at least two sentences;
- batch if officially supported;
- save/load prompt roundtrip if the official data structures are identical to 1.7B's
  (reusing the existing 1.7B voice-clone test patterns);
- sane generated audio.

### 1C. Performance

Extend `scripts/benchmark.py` to `custom-voice-0.6b` and `base-0.6b`. Measure: model
load time; generation wall time; generated audio duration; RTF; peak GPU allocator
usage; VRAM / unified-memory observations where measurable — labeled as observations on
a unified-memory part, not discrete-GPU precision claims. Same methodology as existing
1.7B benchmarks where valid. No subjective 0.6B-vs-1.7B quality comparison.
Evidence: `evidence/benchmark-06b-<date>.{json,txt}`.

### 1D. Documentation

Replace ambiguous "6/6 validated" wording with a per-checkpoint matrix —
Model × (Load / E2E Generate / Benchmark) with ✅/🟡/⬜ states — in `README.md` and
`README_CN.md`, plus a CHANGELOG entry. Equal validation depth is not claimed for all
six repos unless it actually exists.

### Task 1 acceptance criteria

Real GPU tests for 0.6B CustomVoice; real GPU tests for 0.6B Base cloning; real
generated WAV sanity; benchmark evidence for both 0.6B models; docs distinguish load vs
functional validation; previous 1.7B tests still pass; no upstream patch added.

## Task 2 — Multilingual Radeon validation matrix

### 2A. Test data

`tests/data/multilingual_samples.json` — deterministic manifest, one entry per
officially supported language (10 at design time; re-confirmed from
`get_supported_languages()` at runtime — identifiers come from the installed package,
never hardcoded assumptions). Each entry: official language identifier; original native
sentence authored for this repo (no copyrighted prose); expected Unicode text;
capability under test.

### 2B. CustomVoice language matrix

On real GPU, generate at least one sample for every officially supported language via
CustomVoice 1.7B. Validate: API accepts the language; generation completes; waveform
finite; non-silent; valid sample rate; bounded duration. Claim is "end-to-end
generation completed on Radeon" — never pronunciation quality.

### 2C. VoiceDesign multilingual

All 10 languages × one natural-language voice description each (instruction control is
1.7B-supported): generation path executes; instruction accepted; sane waveform. If
upstream support turns out narrower, document the actual coverage.

### 2D. Cross-lingual Voice Clone — representative

Small representative matrix — the default set is exactly 4 pairs: en→zh, zh→en,
ja→en, fr→zh (a pair may be substituted only with a documented reason, e.g. reference
generation failure); reference audio from our own validated generation pipeline.
Explicitly stated as representative, not exhaustive.

### 2E. Machine-readable report

`scripts/validate_languages.py` outputs `evidence/multilingual-matrix.json` +
`evidence/multilingual-matrix.txt` including: timestamp; GPU; gfx; ROCm; torch;
qwen-tts; model alias; language; wall time; audio duration; pass/fail; error if any.
CPU unit tests cover the script's pure logic (manifest validation, aggregation).

### 2F. Documentation

Visible capability matrix — Language × (CustomVoice / VoiceDesign / Clone) — with ✅
only where the evidence file says so, linked to evidence.

### Task 2 acceptance criteria

All current official major languages exercised through at least CustomVoice;
machine-readable evidence; no pronunciation-quality overclaim; representative
cross-lingual clone coverage; docs generated from / traceable to evidence; prior GPU
suite still green; no upstream patches.

## Task 3 — First-class Voice Design → Clone → Reuse workflow

### 3A. Headless backend API

A thin workflow module, `src/qwen3_tts_rocm/voice_workflow.py`, composing only
official APIs:

```
generate_voice_design(text, language, voice description)
    → generated reference audio
create_voice_clone_prompt(ref_audio=generated, ref_text=text)
    → reusable official prompt
generate_voice_clone(text=..., voice_clone_prompt=prompt)
```

No proprietary voice representation. Persistence stores only the official prompt
representation / official-compatible serialization (exact mechanism verified against
the installed package at implementation; existing 1.7B roundtrip knowledge reused).

### 3B. Transcript rule (invariant)

The transcript paired with a generated VoiceDesign reference is exactly the text used
to generate that reference. Enforced by construction and by test. Mismatched ref_text
is never invented.

### 3C. UI

New Gradio tab ("Voice Studio") with the mental model: 1) describe a voice, 2) generate
and hear a preview, 3) save as a named reusable voice, 4) enter new text, 5) generate
with the saved voice. No download → switch tab → re-upload → retype hop. The existing
five tabs remain intact.

### 3D. GPU integration test

VoiceDesign creates sane reference audio; `create_voice_clone_prompt` succeeds using
that result; the reused prompt generates at least two different target sentences, both
sane; serialization roundtrip works; the official prompt object/schema is preserved.

### 3E. Performance observations — three separate phases

VoiceDesign generation time; prompt creation time; reuse generation time — recorded
separately in `evidence/voice-workflow-<date>.{json,txt}` (creation amortizes; reuse
repeats). Never collapsed into one latency number.

### Task 3 acceptance criteria

Real one-click end-to-end workflow; no manual download/re-upload hop; official APIs
only; official-compatible reusable voice data; real GPU integration test; reusable
prompt generates multiple new sentences; evidence archived; standalone VoiceDesign and
Voice Clone paths remain intact.

## Task 4 — ROCm fine-tuning smoke + reproducible reference

**Goal:** prove the official Qwen3-TTS fine-tuning workflow executes end-to-end on the
validated AMD ROCm stack, and document the exact boundary of what was proven. Not
production-quality fine-tuning.

### Execution approach

Upstream `finetuning/` is the official workflow. At execution time, clone
`QwenLM/Qwen3-TTS` at a **pinned SHA into a gitignored working directory** (nothing
upstream is committed into this repo — steady state remains 0 vendored source). The
pinned SHA is recorded in evidence. Official upstream training code is executed; this
repo adds only a thin orchestration layer if strictly necessary. No duplication or
forking of upstream scripts.

### 4A. Minimal legal dataset — self-generated speech

Test dataset generated by this repo's own validated generation pipeline (committed
script + manifest of texts/settings, ~8–16 short utterances across a couple of
languages). No copyrighted or private speech; no unlicensed third-party voice; fully
reproducible. Clearly documented as synthetic engineering-validation data; no quality
claims from it.

### 4B. Data preparation

Official pipeline validated: audio + text + reference → Qwen3-TTS tokenizer → audio
codes → official training dataset format. Evidence archived.

### 4C. Training smoke

Official training script; Base checkpoint (**1.7B Base preferred; 0.6B Base fallback if
memory/time-bound — choice documented in evidence**); deliberately small (~10–50
optimizer steps); ROCm GPU; checkpoint saved successfully. Captured: model; precision;
GPU; ROCm; torch; qwen-tts / upstream commit; batch size; sequence limits; step count;
wall time; peak GPU memory if measurable; final loss values without over-interpreting
them. Purpose is execution validation, not convergence.

### 4D. Reload checkpoint (mandatory)

Training → checkpoint → load checkpoint → synthesize audio → waveform sanity, following
upstream's exact reload path. A training job that merely saves files is not enough.

### 4E. Documentation

`docs/finetuning-rocm.md` distinguishing, plus machine-readable + transcript evidence
at `evidence/finetune-smoke-<date>.{json,txt}`:

**PROVEN:** preprocessing works; training runs N steps; checkpoint saves; checkpoint
reloads; inference produces sane audio.

**NOT PROVEN:** speaker similarity quality; convergence quality; optimal
hyperparameters; long-run stability; multi-speaker training (unless actually tested);
production scalability.

### Task 4 acceptance criteria

Official fine-tuning workflow used; real ROCm training steps; checkpoint artifact
created; checkpoint reloaded; post-finetune inference succeeds; raw evidence archived;
no unsupported quality claim; no upstream patch unless separately justified and
explicitly disclosed.

## Task 5 — Align the repository with the North Star

Only after Tasks 1–4 individually passed verification.

### 5A. README capability table

High-signal section near the top with the four claim states, covering: CustomVoice
1.7B/0.6B; VoiceDesign 1.7B; Voice Clone 1.7B; Base 0.6B; reusable clone prompt;
Design → Clone → Reuse; 12Hz tokenizer; multilingual matrix; fine-tuning; vLLM-Omni
(🚫 roadmap); true streaming (🚫 roadmap). No numeric percentage score unless a stable,
explicitly documented weighting methodology is committed. A factual matrix is
preferred; every green capability links to evidence.

### 5B. Conceptual documentation correctness

Repository-wide audit for capability naming mistakes, specifically downloader
descriptions: CustomVoice is preset-speaker / custom-voice generation — never described
as reference-audio voice cloning. Base is the family providing zero-shot voice cloning
and fine-tuning. Fixed consistently in EN and CN.

### 5C. `patch.py` → `compat.py` evaluation

Evaluate renaming `src/qwen3_tts_rocm/patch.py` (the public promise is zero upstream
patches and `PATCHES` is empty — the name should say what it is). If renaming: preserve
backward compatibility if the module is public; update imports/tests/docs; do not
rename if it introduces avoidable breakage.

### 5D. Validation vocabulary

Repository-wide distinction: "load validated" / "end-to-end validated" / "benchmarked"
/ "not validated". Ambiguous phrases like "all models validated" eliminated unless the
surrounding text defines exactly what validation means.

### 5E. Hardware humility

gfx1151 evidence is never generalized to all Radeon GPUs. Compatibility matrix stays
evidence-driven; community hardware reports remain measured, reproducible reports. No
unsupported multi-GPU claims.

### 5F. Follow-up roadmap (no implementation)

Clearly-labeled roadmap entries: vLLM-Omni on ROCm (feasibility validation first; the
ladder: PyTorch/qwen-tts ROCm → vLLM-Omni offline inference → performance
characterization → online serving when upstream supports it → concurrency → production
guidance); true streaming (required measurements: time to first audio; chunk cadence;
total RTF; buffer underrun behavior; long-text behavior — upstream's "97 ms" never
reused as a Radeon number).

### Task 5 acceptance criteria

North Star visible in README; capability matrix reflects evidence; EN/CN docs
consistent; no CustomVoice/Base conceptual mistakes; no unsupported multi-GPU claims;
"zero upstream patches" still literally true; all prior tests green; every green
capability links to evidence. Final gate = full CPU suite + full GPU suite + zero-patch
audit.

## Final deliverable

After Task 5 PASS: `docs/p0-parity-report.md` following the mission brief's structure —
North Star; repository SHAs (project + upstream); per-task completion with verifier
PASS; capability matrix (Capability × Model × Radeon status × Evidence link);
benchmarks summarized without overstated precision; newly proven Radeon value beyond
upstream; remaining gaps (vLLM-Omni, true streaming, additional Radeon architectures,
long-run fine-tuning, quality evaluation); zero-patch audit; test summary (CPU / GPU /
subagent verification 5/5 PASS); changed files; task commits. Then the **single push**,
and the report summarized to the user.

## Testing philosophy

Tests prove behavior, not implementation trivia: real official objects; public APIs;
waveform sanity; output shape; reproducible runtime behavior; official schema
compatibility; benchmark evidence. Avoid tautological assertions, mock-only tests,
hardcoded internals, assumptions upstream never changes, subjective audio-quality
claims from waveform checks. Stochastic generation: deterministic seeds only to reduce
variance; robust invariants compared; no flaky equality tests; identical seeds are
never mistaken for quality validation.

## Git discipline

One logical commit per independently verified task:

```
Task 0: chore(evidence): record ground-truth audit
Task 1: feat: validate Qwen3-TTS 0.6B generation on Radeon
Task 2: feat: add Radeon multilingual capability matrix
Task 3: feat: add voice-design-to-reusable-voice workflow
Task 4: feat: validate Qwen3-TTS fine-tuning on ROCm
Task 5: docs: align project with capability-first Radeon validation
```

Tasks are never squashed together before independent verification. Verifier FAILs are
fixed inside the same task before moving on. Push happens once, at the end.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subagent cold-start on a hardware-specific stack | Self-contained briefs + accumulated field notes passed forward |
| Context growth over a long run | Durable state lives in commits + `evidence/`; briefs cite files, not conversation |
| GPU flakiness / long generations | Bounded durations, per-language timeouts, failures recorded as failures |
| Fine-tuning memory pressure on gfx1151 | 0.6B Base documented fallback; step count deliberately small |
| Docs drifting from evidence | Verifier claims-audit step every task; matrix cells link evidence files |
| Unified-memory metrics misread as VRAM | Labeled as observations in evidence and docs |

## Out of scope

vLLM-Omni implementation; true streaming implementation; additional Radeon
architectures; long-run fine-tuning; quality evaluation; any modification to upstream
`QwenLM/Qwen3-TTS` or the `qwen-tts` package.
