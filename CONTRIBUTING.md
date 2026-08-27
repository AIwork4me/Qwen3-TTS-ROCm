# Contributing to Qwen3-TTS-ROCm

Thanks for your interest in improving Qwen3-TTS-ROCm! This document explains
how to set up a development environment and the conventions the project uses.

## Project ground rules

This project is deliberately a **thin shim**: it must never fork or patch the
upstream `qwen-tts` source. All synthesis goes through the official public
APIs of the unmodified package. Contributions that add local modifications to
upstream code, or that vendor model weights into the repository, will be
rejected. When in doubt, read the design rationale in
`docs/superpowers/specs/2026-08-27-qwen3-tts-rocm-design.md` (the approved
design doc) — it is the arbiter for "is this in scope?".

The implementation plan lives at
`docs/superpowers/plans/2026-08-27-qwen3-tts-rocm.md`; feature work should
reference where it fits there.

## Getting started

One command sets up everything (venv, pinned AMD ROCm torch wheels, this
package editable with dev+demo extras, GPU sanity gate):

```bash
bash scripts/install.sh                # plus GPU check via scripts/verify_gpu.sh
bash scripts/install.sh --with-models  # also downloads the six checkpoints (~large)
```

Model weights are fetched from official sources only (Hugging Face first,
ModelScope fallback) and live under `models/` — they are never committed.

## Running the tests

Markers are declared in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `gpu` | test needs a working ROCm GPU on this machine |
| `requires_download` | test additionally needs model weights already under `models/` |

Two canonical invocations:

```bash
# CI-class run: fast, CPU-only, no hardware or downloads needed.
python -m pytest -m "not gpu"

# Hardware run: real GPU + downloaded weights (slow, uses the actual stack).
python -m pytest -m "gpu and requires_download" -v
```

If you are touching pure logic (registry, env diagnostics, helpers), a
`pytest -m "not gpu"` pass is sufficient and expected in your PR. If you
change loader/download/demo-backend behavior, also run the hardware suite
when you have access to gfx1151 hardware and state so in the PR.

## Lint / format

Ruff with `line-length = 110` (configured in `pyproject.toml`):

```bash
ruff check .
```

`ruff check` (lint) is enforced locally and in CI and must be clean before
you push. `ruff format` is recommended for new code but is not enforced
repo-wide as of v0.1.0.

## Commit style

Conventional Commits (`feat(scope): ...`, `fix: ...`, `test: ...`,
`docs: ...`, `perf: ...`, `chore: ...`). Commit messages in English or
Chinese are both fine; mixed is acceptable too — keep one style per commit.

## Pull requests

Before opening a PR:

1. `python -m pytest -m "not gpu"` passes;
2. `ruff check .` passes;
3. the behavior follows the zero-modification guarantee above (official API
   calls only);
4. new user-visible behavior is covered by tests;
5. large changes reference the relevant section of the design doc or plan
   under `docs/superpowers/`.

Hardware-dependent changes should paste a short transcript of the
`gpu and requires_download` suite run into the PR description (no binaries,
no `.wav` outputs — text logs only).

## Licensing

By contributing, you agree that your contributions are licensed under the
Apache License 2.0 (see `LICENSE`), consistent with the repository's stated
copyright (`Copyright 2026 Qwen3-TTS-ROCm contributors`). Model weights stay
under Alibaba's own Qwen model license and are never redistributed here.
