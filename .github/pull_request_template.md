<!--
Thank you for contributing to Qwen3-TTS-ROCm!
Ground rule: this project is a thin shim around the unmodified official
qwen-tts package — see CONTRIBUTING.md.
-->

## What changed

<!-- One or two sentences. -->

## Why

<!-- Link the issue ("Fixes #123") or describe the motivation. -->

## Validation

<!-- Paste real command output, not "should work". -->

```text
# ruff check .
# python -m pytest -m "not gpu and not requires_download" -q
# (where applicable) python -m pytest -m "gpu" -q
```

## Hardware

<!-- e.g. validated on Radeon 8060S / gfx1151 / ROCm 7.14.0, or "CPU-only change — not hardware-tested" -->

## Evidence

<!-- For new compatibility/performance claims: link the artifact in evidence/, the issue form, or the benchmark JSON. -->

## Checklist

- [ ] I did not vendor or patch upstream `qwen-tts` source
- [ ] CPU tests pass (`pytest -m "not gpu and not requires_download"`)
- [ ] GPU behavior was tested where applicable
- [ ] Documentation was updated (EN + CN where user-facing)
- [ ] New compatibility claims include evidence
- [ ] No model weights are committed
