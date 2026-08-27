# Qwen3-TTS-ROCm

ROCm (gfx1151) runtime adapter, model manager and enhanced Gradio demo for Qwen3-TTS.

## Status

Work in progress on branch `feat/impl-v0.1.0`. See `docs/` for the design and implementation plan.

## Install (development)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Note: PyTorch ROCm wheels are installed separately from the AMD index; this package intentionally declares no torch dependencies.
