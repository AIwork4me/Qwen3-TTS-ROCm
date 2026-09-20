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

## [Unreleased]

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
