#!/usr/bin/env bash
# run_demo.sh — launch the enhanced Gradio demo (implemented in Tasks 15/16).
#
# Usage:
#   bash scripts/run_demo.sh [--port N] [any extra args are forwarded verbatim]
#
# Default port: 8000.  The script activates .venv and execs the
# `qwen3-tts-rocm-demo` console entry point, forwarding all arguments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEMO="$ROOT/.venv/bin/qwen3-tts-rocm-demo"
ACTIVATE="$ROOT/.venv/bin/activate"
if [[ ! -x "$DEMO" || ! -f "$ACTIVATE" ]]; then
    echo "ERROR: .venv entry points missing — run scripts/install.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$ACTIVATE"
exec qwen3-tts-rocm-demo "$@"
