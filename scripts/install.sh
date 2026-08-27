#!/usr/bin/env bash
# install.sh — one-command setup for Qwen3-TTS-ROCm.
#
# Usage:
#   bash scripts/install.sh [--with-models]
#
# What it does (idempotent — safe to re-run):
#   1. creates .venv (prefers uv, falls back to python3 -m venv) unless present;
#   2. installs the pinned AMD ROCm torch wheel stack from the AMD index;
#   3. installs this package in editable mode with dev extras (-e ".[dev]");
#   4. with --with-models, runs scripts/download_models.sh afterwards;
#   5. finishes with scripts/verify_gpu.sh, printing SPIKE-GPU-OK on success.
#
# uv fallback note: python3 -m venv bootstraps pip through ensurepip; we then
# upgrade pip explicitly so both paths converge on the same tooling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=1 ;;
        -h|--help)
            sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$arg' (只支持 --with-models)" >&2
            exit 2
            ;;
    esac
done

# --- 1. virtualenv ----------------------------------------------------------
UV_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
fi

if [[ -x ".venv/bin/python" ]]; then
    echo "==> [venv] .venv already exists — skipping creation"
elif [[ -n "$UV_BIN" ]]; then
    echo "==> [venv] creating .venv with uv ($UV_BIN)"
    "$UV_BIN" venv --seed .venv
else
    echo "==> [venv] uv not found — falling back to python3 -m venv (+ensurepip/upgrade pip)"
    python3 -m venv .venv
    .venv/bin/python -m ensurepip --upgrade
fi

PY=".venv/bin/python"

# --- 2. pinned AMD ROCm torch stack ----------------------------------------
# EXACT command from Global Constraints (do not edit versions or index):
echo "==> [torch] pinned AMD ROCm wheels from repo.amd.com (skipped fast when satisfied)"
.venv/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0" --no-input

# --- 3. this package, editable, with dev extras -----------------------------
echo "==> [package] pip install -e \".[dev]\""
$PY -m pip install --no-input -e ".[dev]"

# --- 4. optional model fetch -------------------------------------------------
if (( WITH_MODELS )); then
    echo "==> [models] --with-models requested — running scripts/download_models.sh"
    bash "$ROOT/scripts/download_models.sh"
fi

# --- 5. GPU sanity gate (prints SPIKE-GPU-OK) --------------------------------
echo "==> [verify] running scripts/verify_gpu.sh"
bash "$ROOT/scripts/verify_gpu.sh"
