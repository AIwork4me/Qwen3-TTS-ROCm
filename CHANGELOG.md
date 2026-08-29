# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
