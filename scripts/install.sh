#!/usr/bin/env bash
# install.sh — one-command setup for Qwen3-TTS-ROCm.
#
# Usage:
#   bash scripts/install.sh [--gfx-target {gfx1100,gfx1151,auto}] [--with-models]
#
# Architecture contract (docs/development/multiarch-design.md):
#   --gfx-target gfx1100  install the validated gfx1100 wheel extras
#                         (device-gfx1100; Radeon Pro W7900D class, ROCm 7.14)
#   --gfx-target gfx1151  install the validated gfx1151 wheel extras
#                         (device-gfx1151; Radeon 8060S class, ROCm 7.14)
#   --gfx-target auto     DEFAULT. Fail-closed auto-detection: enumerate the
#                         gfx arch of every visible compute agent
#                         (rocm_agent_enumerator, falling back to rocminfo);
#                         proceed ONLY when exactly one distinct known arch
#                         is visible. No tools / no GPU / mixed archs /
#                         unknown arch => hard error telling you to pass
#                         --gfx-target explicitly. Never guesses, never
#                         accepts a family pattern like gfx11*, never sets
#                         HSA_OVERRIDE_*.
#
# What it does (idempotent — safe to re-run):
#   1. resolves the gfx target (fail-closed, above) BEFORE any download;
#   2. creates .venv (prefers uv, falls back to python3 -m venv) unless
#      present;
#   3. installs the pinned AMD ROCm torch wheel stack from the AMD index,
#      with the device extras for the selected architecture;
#   4. installs this package in editable mode with the dev+demo extras
#   (-e ".[dev,demo]");
#   5. with --with-models, runs scripts/download_models.sh afterwards;
#   6. finishes with scripts/verify_gpu.sh, printing SPIKE-GPU-OK on success.
#
# uv fallback note: python3 -m venv bootstraps pip through ensurepip; we then
# upgrade pip explicitly so both paths converge on the same tooling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MODELS=0
GFX_TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-models) WITH_MODELS=1; shift ;;
        --gfx-target)
            [[ $# -ge 2 ]] || { echo "ERROR: --gfx-target requires a value (gfx1100 | gfx1151 | auto)" >&2; exit 2; }
            GFX_TARGET="$2"; shift 2 ;;
        --gfx-target=*)
            GFX_TARGET="${1#--gfx-target=}"; shift ;;
        -h|--help)
            sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1' (supported: --gfx-target {gfx1100,gfx1151,auto}, --with-models)" >&2
            exit 2
            ;;
    esac
done
[[ -z "$GFX_TARGET" ]] && GFX_TARGET="auto"

# --- 0. architecture contract (docs/development/multiarch-design.md) ---------
# Resolved BEFORE the venv exists and long before the multi-GB wheel
# downloads. Unknown/ambiguous input fails immediately; there is no
# family-pattern shortcut and no fallback to CUDA/PyPI wheels, ever.
# >>> gfx-target-table (tests/test_install_sh.py pins this block)
gfx_extras_for() {
    case "$1" in
        gfx1100) echo "device-gfx1100" ;;
        gfx1151) echo "device-gfx1151" ;;
        *) return 1 ;;
    esac
}

detect_visible_gfx_archs() {
    # Print every distinct gfx arch of visible compute agents, one per line.
    # Probe order per the design doc: rocm_agent_enumerator, then rocminfo.
    # Only exact `gfx<4 digits>` tokens count — ISA strings such as
    # amdgcn-amd-amdhsa--gfx1100 or gfx11-generic can never match.
    local out rc
    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        out="$(rocm_agent_enumerator 2>/dev/null || true)"
    elif command -v rocminfo >/dev/null 2>&1; then
        out="$(rocminfo 2>/dev/null | awk '/^ *Name: */{print $2}' || true)"
    else
        return 3  # no ROCm userspace tool available
    fi
    local archs
    archs="$(printf '%s\n' "$out" | tr '[:space:]' '\n' | grep -E '^gfx[0-9]{4}$' || true)"
    [[ -z "$archs" ]] && return 1  # tools present, no visible compute agent
    printf '%s\n' "$archs" | sort -u
}

if [[ "$GFX_TARGET" == "auto" ]]; then
    det=""; det_rc=0
    det="$(detect_visible_gfx_archs)" || det_rc=$?
    mapfile -t det_list <<< "$det"
    if [[ $det_rc -ne 0 || ${#det_list[@]} -ne 1 ]]; then
        echo "ERROR: --gfx-target auto could not resolve exactly one visible gfx architecture" >&2
        [[ $det_rc -eq 3 ]] && echo "  (no rocm_agent_enumerator/rocminfo on PATH — install ROCm userspace or pass the target explicitly)" >&2
        [[ $det_rc -eq 1 ]] && echo "  (no visible compute agent — is a GPU present and /dev/kfd accessible?)" >&2
        [[ $det_rc -eq 0 ]] && { echo "  (multiple distinct architectures visible — ambiguous:" >&2; printf '   %s\n' "${det_list[@]}" >&2; } || true
        echo "  Fix: bash scripts/install.sh --gfx-target gfx1100   (or gfx1151)" >&2
        exit 1
    fi
    GFX_TARGET="${det_list[0]}"
    if ! gfx_extras_for "$GFX_TARGET" >/dev/null; then
        echo "ERROR: visible architecture '${GFX_TARGET}' is not a validated target of this project (supported: gfx1100, gfx1151)" >&2
        echo "  This is NOT a claim that your GPU is unsupported — it only has no validated wheel contract here yet." >&2
        exit 1
    fi
    echo "==> [gfx] auto-detected target: ${GFX_TARGET} (single distinct visible arch)"
fi

GFX_EXTRAS="$(gfx_extras_for "$GFX_TARGET")" || {
    echo "ERROR: invalid --gfx-target '${GFX_TARGET}' (supported: gfx1100, gfx1151, auto)" >&2
    exit 2
}
echo "==> [gfx] target: ${GFX_TARGET} | wheel extras: ${GFX_EXTRAS}"
# <<< gfx-target-table

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

# --- python guard: fail fast BEFORE the multi-GB ROCm wheel downloads -------
# >>> python-version-guard (tests/test_install_sh.py re-runs this exact block)
"$PY" - <<'PY'
import sys

MIN = (3, 10)
v = sys.version_info
if v < MIN:
    raise SystemExit(
        f"ERROR: Python {MIN[0]}.{MIN[1]}+ is required (需要 Python "
        f"{MIN[0]}.{MIN[1]}+); detected {v.major}.{v.minor}.{v.micro}."
    )
print(f"==> [python] Python {v.major}.{v.minor}.{v.micro}")
PY
# <<< python-version-guard

# --- 2. pinned AMD ROCm torch stack ----------------------------------------
# EXACT command shape from Global Constraints (do not edit versions or index):
# only the device extras vary, per the architecture contract resolved above.
# gfx1151 => torch[device-gfx1151]/torchvision[device-gfx1151] (v0.2.1 command,
# byte-identical); gfx1100 => torch[device-gfx1100]/torchvision[device-gfx1100]
# (issue #1 validated command). torchaudio ships no device extra on this index.
echo "==> [torch] pinned AMD ROCm wheels from repo.amd.com for ${GFX_TARGET} (skipped fast when satisfied)"
.venv/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[${GFX_EXTRAS}]==2.12.0+rocm7.14.0" "torchvision[${GFX_EXTRAS}]==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0" --no-input

# --- 3. this package, editable, with dev+demo extras ------------------------
# UX-fix U2: pip prints nothing for long stretches while unpacking the wheel
# stack — warn the user up front so silence is not mistaken for a hang.
echo "==> [3/4] installing project + deps (large ROCm wheels; this may take a few minutes — 较大的 ROCm wheel 解压可能需要几分钟，若长时间无输出属正常)"
echo "==> [package] pip install -e \".[dev,demo]\""
$PY -m pip install --no-input -e ".[dev,demo]"

# --- 4. optional model fetch -------------------------------------------------
if (( WITH_MODELS )); then
    echo "==> [models] --with-models requested — running scripts/download_models.sh"
    bash "$ROOT/scripts/download_models.sh"
fi

# --- 5. GPU sanity gate (prints SPIKE-GPU-OK) --------------------------------
# QWEN3_TTS_ROCM_SKIP_VERIFY=1 opts out for CONTAINER IMAGE BUILDS: there is no
# /dev/kfd inside `docker build`, yet the image must pin the IDENTICAL wheel
# stack via this script (single source of truth; see docker/Dockerfile). The
# default (unset/0) keeps the gate ON for every bare-metal host.
if [[ "${QWEN3_TTS_ROCM_SKIP_VERIFY:-0}" == "1" ]]; then
    echo "==> [verify] skipped (QWEN3_TTS_ROCM_SKIP_VERIFY=1 — container build)"
else
    echo "==> [verify] running scripts/verify_gpu.sh"
    bash "$ROOT/scripts/verify_gpu.sh"
fi

# UX-fix U2: without weights the demo loads nothing — make the next step the
# last line the user sees on a successful install.
echo "NEXT: bash scripts/download_models.sh   # six repos ≈18GB (或指定子集: download_models.sh tokenizer custom-voice)"
