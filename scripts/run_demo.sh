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

# --- UX-fix U2: startup honesty about missing models -------------------------
# Crudely recover the model reference the way the CLI would: `--alias X`
# (or `-c/--checkpoint X`, or `--alias=X`), else the first positional that is
# neither a flag nor a flag value; default custom-voice.  The server STILL
# starts (loading is lazy) — this WARN only sets expectations before the UI
# comes up, instead of a failure surfacing mid-generation.
_DEMO_ALIAS=""
_prev=""
for _tok in "$@"; do
    if [[ "$_prev" =~ ^(--alias|-c|--checkpoint)$ ]]; then
        _DEMO_ALIAS="$_tok"                       # value of an alias-ish flag
    elif [[ "$_tok" == --alias=* ]]; then
        _DEMO_ALIAS="${_tok#--alias=}"
    elif [[ "$_tok" != -* && ! "$_prev" =~ ^(--models-dir|--device|--dtype|--ip|--port|--concurrency|--ssl-certfile|--ssl-keyfile|--max-new-tokens|--temperature|--top-k|--top-p|--repetition-penalty|--subtalker-top-k|--subtalker-top-p|--subtalker-temperature)$ ]]; then
        _DEMO_ALIAS="${_DEMO_ALIAS:-$_tok}"       # first real positional
        break
    fi
    _prev="$_tok"
done
_DEMO_ALIAS="${_DEMO_ALIAS:-custom-voice}"

_PY="$ROOT/.venv/bin/python"
if _MODEL_DIR="$("$_PY" -c '
import sys

from qwen3_tts_rocm import models

try:
    print(models.local_dir(sys.argv[1]))
except KeyError:
    sys.exit(3)  # not a registry alias (local path / repo id): loader reports it later
' "$_DEMO_ALIAS" 2>/dev/null)"; then
    if ! "$_PY" -c '
import sys

from qwen3_tts_rocm import models

sys.exit(0 if models.is_downloaded(sys.argv[1], require_weights=True) else 1)
' "$_MODEL_DIR"; then
        echo "WARN: model '${_DEMO_ALIAS}' not found in ${_MODEL_DIR} — generation will fail until you run: bash scripts/download_models.sh ${_DEMO_ALIAS}" >&2
    fi
fi

# shellcheck source=/dev/null
source "$ACTIVATE"
exec qwen3-tts-rocm-demo "$@"
