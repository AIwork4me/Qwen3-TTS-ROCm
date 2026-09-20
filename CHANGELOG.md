# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-20

### Added

- 0.6B end-to-end capability validation on the Radeon 8060S (P0 capability
  parity, Task 1): new on-GPU test suites for `custom-voice-0.6b`
  (metadata surface, single/batch generation, sampling-kwarg passthrough,
  and the pinned `instruct` boundary — accepted but silently ignored by the
  installed wrapper for 0.6B, per the Task 0 ground-truth audit) and
  `base-0.6b` (ref-text cloning, x-vector-only cloning, reusable
  `create_voice_clone_prompt`, official-demo-format save/load roundtrip
  parity, batch cloning). The suite grows to 250/250 on the validation host
  (216 CPU + 34 real-GPU); CPU-only CI passes 215 + 1 HIP-gated skip.
- `scripts/benchmark.py`: the 0.6B aliases (`custom-voice-0.6b`,
  `base-0.6b`) are benchmarked through their family's official entry
  points; per-alias `load_seconds` (timed `loader.load`) and
  `peak_alloc_gb` (torch peak allocation across the alias's run, reset
  after load) are printed, stored under `meta.per_alias`, and attached to
  every per-cell record; JSON `meta` now also records `git_head` and the
  audited upstream `upstream_qwen3_tts_sha`. Archived 0.6B evidence run:
  `evidence/benchmark-06b-2026-09-20.json` + verbatim transcript
  `evidence/benchmark-06b-2026-09-20.txt`.
- README / README_CN: per-checkpoint validation-level matrix (Load /
  E2E Generate / Benchmark) directly under the verified-results table,
  separating load validation from functional validation; the model-repo
  row now reads "6 / 6 load-validated"; documented boundary note that 0.6B
  CustomVoice has no instruction control upstream (the installed wrapper
  accepts and nulls `instruct` for 0.6B).
- Multilingual Radeon validation matrix (P0 capability parity, Task 2):
  new harness `scripts/validate_languages.py` + authored fixture
  `tests/data/multilingual_samples.json` exercising every officially
  supported language end to end on the Radeon GPU — 10 CustomVoice + 10
  VoiceDesign cells + 4 representative cross-lingual clone pairs (1.7B
  models, one resident at a time, every generation keyword-first with
  `max_new_tokens=512`). Manifest language names resolve case-insensitively
  onto the installed package's `get_supported_languages()` identifiers
  (recorded per row); drift in either direction is a hard error, never a
  silent skip. The only capability claim is "end-to-end generation
  completed on Radeon" from waveform sanity (finite, non-silent, valid
  sample rate, bounded duration) — no pronunciation-quality claims
  anywhere; cross-lingual clone coverage is stated as representative
  (4 pairs), not exhaustive. The cross-lingual transcript rule is
  structural: `ref_text` is exactly the manifest sentence synthesized into
  the reference audio. 13 CPU unit tests pin the pure helpers. Archived
  run: `evidence/multilingual-matrix.json` + verbatim transcript
  `evidence/multilingual-matrix.txt` (24/24 matrix cells + 4/4 reference
  generations passed, 147.5 s wall). README / README_CN gained the
  "Multilingual capability matrix" section rendered from that evidence.
- First-class Voice Design → reusable-voice workflow (P0 capability parity,
  Task 3): new `qwen3_tts_rocm.voice_workflow` composes ONLY the three
  official inference APIs (`generate_voice_design`,
  `create_voice_clone_prompt`, `generate_voice_clone`) on unmodified
  `loader.load` objects into describe → preview → save → reuse, honouring
  the official model split (the wrapper hard-gates design on the
  VoiceDesign checkpoint and prompt/reuse on Base — probe-proven, so
  `design_voice` takes the Base object as `prompt_model`). The transcript
  rule is enforced by construction: one `text` local feeds both the
  generation and the `ref_text`. Saved voices are official-demo-compatible
  `.pt` files (the `"items"` key is byte-format-identical to upstream's
  `{"items": [asdict(item) ...]}` payload) plus an in-file `voice_meta`
  sidecar that survives `torch.load(..., weights_only=True)` on the
  installed torch 2.12 (the pinned persistence variant; no sibling .json
  needed). The demo gained the ⑥ Voice Studio (音色工坊) tab — one-click
  design/preview/save/reuse with NO download/re-upload hop (the
  saved-voices dropdown replaces the file round-trip; tabs ①–⑤ untouched) —
  backed by new `SynthesisService.voice_studio_design` /
  `voice_studio_save` / `voice_studio_list` / `voice_studio_generate`
  methods (official split preserved through the size-1 model LRU: the
  VoiceDesign weights are evicted exactly when the prompt phase begins).
  `FakeTTSModel`'s prompt-item double gained the official fifth field
  `ref_text` (schema-faithful mirror). Suite grows to 282/282 on the
  validation host (244 CPU + 38 real-GPU: +15 CPU demo/backend/UI tests,
  +4 GPU workflow tests). Archived evidence with the three phases timed
  SEPARATELY: `evidence/voice-workflow-2026-09-20.json` (design 5.47 s,
  prompt 0.31 s, reuse 5.28 s / 6.42 s; every generation
  `max_new_tokens=512`) + verbatim transcript
  `evidence/voice-workflow-2026-09-20.txt`. README / README_CN gained the
  "Voice Design → reusable voice (Voice Studio)" section.
- Fine-tuning execution-validation smoke on ROCm (P0 capability parity,
  Task 4): the OFFICIAL upstream `finetuning/` workflow
  (`prepare_data.py` → `sft_12hz.py` → per-epoch checkpoint →
  `Qwen3TTSModel.from_pretrained` reload → `generate_custom_voice`)
  proven to execute end-to-end on the Radeon GPU. New
  `scripts/make_finetune_dataset.py` + authored manifest
  `tests/data/finetune_manifest.json` build a SELF-GENERATED corpus
  (12 original utterances — 6 Chinese + 6 English, distinct from the
  multilingual samples — plus one shared reference clip, all rendered by
  the official 1.7B CustomVoice model with `torch.manual_seed(1234)`
  before each render and `max_new_tokens=512`; transcript == manifest
  text by construction) in exactly upstream's JSONL schema; 8 CPU tests
  pin its pure helpers. Training smoke: 1.7B Base bf16, batch 2 / lr 2e-5
  (upstream defaults), 2 epochs = 12 optimizer steps (upstream steps per
  microbatch), wall 19.0 s, peak `torch.cuda.max_memory_allocated`
  18.02 GiB, loss lines quoted verbatim with no interpretation; reload
  synthesis of a manifest sentence is finite, 24 kHz, 3.52 s, non-silent
  (`assert_wav_sane` PASS). Scope is EXECUTION VALIDATION ONLY — no
  speaker-similarity / convergence / hyperparameter / stability /
  multi-speaker / scalability claims anywhere (NOT-PROVEN list in the
  doc). Upstream defects + deviations disclosed: (1) `sft_12hz.py`
  hard-codes `flash_attention_2`, which fails at init on the
  flash-attn-less ROCm stack (verbatim ImportError captured); a
  single-line `sdpa`
  override was applied INSIDE the gitignored `.upstream` clone, diff
  recorded, clone restored pristine afterwards (`git status --porcelain`
  empty at the pinned SHA — zero-patch audit); upstream issue filing
  noted as a human-owner action. (2) `github.com:443` was unreachable at
  run time, so the pinned clone was bootstrapped from codeload + the
  GitHub API with the commit object reconstructed SHA-identically to the
  pinned upstream SHA (authenticity by git content-addressing; procedure
  in the transcript). Scratch lives in gitignored `.upstream/` and
  `.work-finetune/` (plus `voices/`); NOTHING upstream or checkpoint
  shaped enters the commit. Archived evidence:
  `evidence/finetune-smoke-2026-09-20.json` + phase-marked verbatim
  transcript `evidence/finetune-smoke-2026-09-20.txt`; new reference doc
  `docs/finetuning-rocm.md` (PROVEN/NOT-PROVEN lists, reproduction
  commands); README / README_CN capability rows mark fine-tuning as
  ✅-scoped execution-only.
- Repository alignment with the North Star (P0 capability parity, Task 5):
  README / README_CN now open with the North Star sentence ("Official
  Qwen3-TTS on AMD Radeon — capability by capability, benchmark by
  benchmark, with zero upstream patches." / CN mirror) followed by a
  consolidated four-state capability matrix (✅ Radeon E2E validated · 🟡
  partial-load-only · ⬜ not validated · 🚫 not exposed upstream or
  intentionally not claimed) covering CustomVoice 1.7B/0.6B, VoiceDesign
  1.7B, Voice Clone 1.7B, Base 0.6B, reusable clone prompts,
  Design→Clone→Reuse, the 12Hz tokenizer, the multilingual matrix,
  fine-tuning (execution-only smoke), vLLM-Omni (🚫 roadmap) and true
  streaming (🚫 roadmap) — every ✅ cell links its evidence file, no
  percentage scores; the existing per-checkpoint and multilingual matrices
  kept as detail views. Conceptual boundary fixed in
  `scripts/download_models.sh` help (CustomVoice was misfiled as
  "reference-voice cloning": it is preset/custom-speaker generation; Base
  is the zero-shot-cloning + fine-tuning family) and stated explicitly in
  both READMEs. Stale suite counts recomputed honestly and identically in
  EN/CN: validation host 252 CPU + 38 GPU = 290/290; the CPU-only CI
  matrix runs the same 252 CPU tests with 1 HIP-gated skip (previously
  contradictory 215/243/216/244/34 figures). A clearly-labelled
  "Roadmap (not yet validated)" section was added to both READMEs: the
  vLLM-Omni-on-ROCm ladder (feasibility first, online serving only when
  upstream supports the required path, no concurrency assumptions on the
  single-GPU iGPU) and true streaming with its required measurements
  (time to first audio, chunk cadence, total RTF, buffer underrun,
  long-text behaviour), noting explicitly that upstream's "97 ms" figure
  is NOT a Radeon number. `src/qwen3_tts_rocm/patch.py` renamed to
  `compat.py` (module docstring now describes what it actually is — a
  version advisory plus a deliberately-empty reserved patch point; the
  old name overstated it) with all internal imports/tests updated and no
  compatibility shim (the only importers were `loader.py` and the test
  suite); full CPU + GPU suites re-run green after the rename.
  `evidence/README.md` gained an errata note for two misleading labels in
  the fine-tuning transcript (line 108's "pristine" label follows a
  mid-bootstrap staged listing; line 1546's `as-is-run-exit=0` is the
  `| tail -40` pipe's exit, not the failed run's) and its
  multilingual-transcript row no longer mentions a `RUN_EXIT=0` token the
  transcript does not contain — transcripts themselves stay byte-for-byte
  unedited per the no-hand-edit policy. Task 1–4 verifier reports copied
  verbatim to `docs/superpowers/reports/`; final program report at
  `docs/p0-parity-report.md`. Two latent `ruff` errors in
  `scripts/make_finetune_dataset.py` fixed (missing executable bit,
  unparenthesized implicit concatenation) so CI's lint job passes.

## [Unreleased]

### Verified

- Filed upstream issue [QwenLM/Qwen3-TTS#372](https://github.com/QwenLM/Qwen3-TTS/issues/372):
  `finetuning/sft_12hz.py` hard-codes `attn_implementation="flash_attention_2"`, failing
  out-of-the-box on ROCm; requests a CLI-selectable attention implementation or an
  `sdpa` fallback (`evidence/upstream-issue-2026-09-21.txt`).
- First verified CI run on the 0.2.0 suite: push `8815238`, run
  35526426415 — `test (3.10/3.11/3.12)` + `build` all green,
  `251 passed, 1 skipped, 38 deselected` per Python job
  (`evidence/ci-2026-09-21-8815238.txt`); README/README_CN/P0 report now
  cite the run instead of a host-derived expectation.

### Changed

- Claims-consistency round: runtime messages no longer make claims the
  project cannot back — the first-load announcement describes cache/kernel
  causes instead of predicting "tens of seconds", and the flash-attn guard
  error (EN+CN) is scoped to the validated wheel stack. CPU CI now runs a
  real Python 3.10 / 3.11 / 3.12 matrix (locally verified on all three
  before authoring), split into test/build jobs with `permissions:
  contents: read` and actions pinned to full commit SHAs. `install.sh`
  fails fast on an unsupported interpreter before the multi-GB ROCm wheel
  downloads (covered by `tests/test_install_sh.py`) and no longer quotes a
  drift-prone wheel size. The hardware-validation issue form warns
  non-gfx1151 testers away from the gfx1151-specific installer and now
  requires exact install commands; Docker size wording names the metric
  behind every number (inspect `.Size` / `docker history` layer /
  `docker system df -v`).

- Public-facing trust pass (EN+CN kept in lockstep): READMEs restructured
  hero-first — verified-results table up top, a minimal ~5 GB quick start
  before the full ~18 GB download, a compatibility matrix that separates the
  validated gfx1151 configuration from unvalidated hardware, collapsed
  five-tab screenshots and a curated generated audio sample linked from the
  hero. Internal development documents (design spec, implementation plan,
  acceptance record) consolidated under `docs/development/`; superseded
  Docker logs and an internal test-infra note removed from `evidence/`, with
  `evidence/README.md` added as a reproducibility index. Stale pre-remote
  notes removed from the CI workflow header; `pyproject.toml` gained
  keywords, classifiers and project URLs. GPU suite re-run green on the
  validation host after the changes (transcript:
  `evidence/gpu-suite-2026-08-29.txt`).

- Documentation and UX pass: the README was restructured with the Quickstart
  moved up front, plus a per-repository model size table, subset-download
  instructions and documented first-run log expectations; the minimal usage
  snippet now saves a wav file; `--help` output no longer renders doubled
  BooleanOptional flags.

### Fixed

- Per-tab automatic model routing: generating from any demo tab now
  auto-selects the matching model family, so a mismatched sidebar switcher
  selection can no longer produce "does not support generate_custom_voice"
  errors — a switch notice is shown instead.
- First-run terminal noise: upstream import banners (SoX, flash-attn) are
  suppressed at the file-descriptor level and sdpa experimental warnings are
  filtered, while the loader prints a bilingual first-load expectation line;
  escape hatches are `QWEN3_TTS_ROCM_VERBOSE_IMPORT=1` and
  `QWEN3_TTS_ROCM_QUIET=1`.
- CLI and script friction: `scripts/download_models.sh` gained a usage line
  and clean unknown-alias errors, `scripts/run_demo.sh` warns when models
  are missing before launch, a busy port yields a bilingual hint suggesting
  `--port`, loader/download/refusal diagnostics carry troubleshooting
  pointers, the startup banner is humanized and `scripts/install.sh` prints
  stage banners with a NEXT hint.

## [0.1.0] - 2026-08-27

First public-ready release: a thin, zero-modification shim that runs the
official `qwen-tts` package on AMD ROCm 7.14.0 (gfx1151), with an enhanced
bilingual demo, GPU test suite and measured performance numbers.

### Added

- ROCm 7.14.0 gfx1151 support shim:
  - environment diagnostics with a never-raise guarantee, surfaced through
    the `qwen3-tts-rocm-check` CLI;
  - smart-default model loader (`loader.load()` / `loader.unload()`) that
    returns the native official Qwen3-TTS model object — all synthesis stays
    on unmodified official APIs — with a metadata-backed version probe and a
    silent clean refusal for unsupported combos;
  - dual-source model downloader (ModelScope first, hf-mirror.com fallback,
    per-alias auto mode) with friendly registry aliases for all six official
    checkpoints.
- One-command setup and launch scripts: `scripts/install.sh`
  (`--with-models` optional), `scripts/download_models.sh`,
  `scripts/run_demo.sh`, `scripts/verify_gpu.sh`.
- Enhanced bilingual (中文/English) Gradio demo application with five tabs:
  voice clone (incl. a save/load voice sub-tab), preset speakers, voice
  design, codec (encode→decode roundtrip visualization through the official
  12Hz speech tokenizer) and generation history — backed by a headless
  `SynthesisService` with server-side file validation.
- pytest suite covering unit logic plus on-GPU integration: all-model
  load/unload smoke, tokenizer encode/decode roundtrip, full custom-voice
  matrix, voice design, voice clone workflow (incl. prompt reuse and
  save/load) and waveform sanity helpers.
- Upstream parity proof: GPU integration test demonstrating that the
  upstream demo runs unmodified on this stack (thin-shim guarantee).
- Reproducible RTF benchmark tool (`scripts/benchmark.py`) with published
  methodology; results interpreted in `docs/benchmarks.md` with raw session
  and JSON evidence under `evidence/`.
- Docker image recipe and a CPU-only GitHub Actions CI pipeline
  (ruff + `pytest -m "not gpu"`); container exposes `/dev/kfd`, `/dev/dri`
  device passthrough for ROCm execution.

### Performance

- Measured real-time factors on gfx1151 (Ryzen AI Max+ PRO 395 iGPU,
  bfloat16/sdpa): tuned voices render short/medium sentences at a median
  RTF of 1.3–1.6 (lower is better); zero-shot voice cloning measured
  around RTF ~1.8. Full per-cell tables and caveats in `docs/benchmarks.md`.

### Security

- Model downloads are restricted to the two official registries with pinned
  repository ids; no third-party mirrors, no weights in the repo or wheel.
- Environment probing is strictly read-only and never raises (fail-safe by
  design) — hardware discovery failures degrade to plain reports instead of
  stack traces or partial loads.
- Demo uploads are validated on the backend before use (moved out of the UI
  layer so the guarantee holds for API clients too).
