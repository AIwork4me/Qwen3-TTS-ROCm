#!/usr/bin/env bash
# verify_gpu.sh — quick ROCm/GPU sanity check for the Qwen3-TTS-ROCm project.
# Validates that the AMD pip-wheel torch stack sees a project-validated AMD
# GPU (gfx1100 or gfx1151; exact-match contract — see
# docs/development/multiarch-design.md) and can run bf16 matmul + SDPA +
# torchaudio import. Exits non-zero on any failure.
#
# Usage (run from repo root, expects .venv):
#   bash scripts/verify_gpu.sh                       # target auto (default)
#   QWEN3_TTS_ROCM_GFX_TARGET=gfx1100 bash scripts/verify_gpu.sh
#   QWEN3_TTS_ROCM_GFX_TARGET=gfx1151 bash scripts/verify_gpu.sh
#
# Semantics:
#   - auto     : the ACTUAL arch (torch gcnArchName, ground truth) must be
#                exactly one of the validated targets; anything else fails
#                as "ROCm-visible but not validated architecture".
#   - gfxNNNN  : the actual arch must EQUAL the request; a mismatch is a
#                hard failure (never reinterpreted).
#   - invalid values exit 2 before touching the GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README.md Quick Start)." >&2
    exit 1
fi

"$PY" - <<'EOF'
import os
import re
import sys

import torch
import torch.nn.functional as F

from qwen3_tts_rocm import gfx

requested = os.environ.get("QWEN3_TTS_ROCM_GFX_TARGET", "auto").strip().lower()
valid = ("auto", *sorted(gfx.KNOWN_TARGETS))
if requested not in valid:
    print(
        f"ERROR: QWEN3_TTS_ROCM_GFX_TARGET='{requested}' is invalid "
        f"(supported: {', '.join(valid)})",
        file=sys.stderr,
    )
    sys.exit(2)

print("torch", torch.__version__, "| HIP", torch.version.hip)
assert torch.version.hip is not None, "not a ROCm build of torch"
assert torch.version.hip.startswith("7.14"), (
    f"expected HIP 7.14.x, got {torch.version.hip}"
)
assert torch.cuda.is_available(), "cuda not available under ROCm build"

dev = "cuda:0"
props = torch.cuda.get_device_properties(0)
arch = str(getattr(props, "gcnArchName", "") or "")
print("GPU:", props.name, "| arch:", arch or "(missing)")

if not re.fullmatch(r"gfx[0-9]{4}", arch):
    print(
        f"ERROR: cannot determine an exact gfx arch (got {arch!r}) — "
        "no family-pattern matching is performed",
        file=sys.stderr,
    )
    sys.exit(1)

if requested == "auto":
    if gfx.classify_arch(arch) is None:
        print(
            f"ERROR: {gfx.classification_line(arch)} — refusing to call it "
            "validated; pass an explicit QWEN3_TTS_ROCM_GFX_TARGET only for "
            "a validated target",
            file=sys.stderr,
        )
        sys.exit(1)
else:
    if arch != requested:
        print(
            f"ERROR: architecture mismatch — requested target '{requested}' "
            f"but the visible GPU is '{arch}' (exact match required; this is "
            "a configuration error, not a validation)",
            file=sys.stderr,
        )
        sys.exit(1)

print(gfx.classification_line(arch))

a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
b = a @ a
assert torch.isfinite(b).all().item(), "bf16 matmul produced non-finite values"

q = torch.randn(4, 8, 128, device=dev, dtype=torch.bfloat16)
o = F.scaled_dot_product_attention(q, q, q)
assert torch.isfinite(o).all().item(), "SDPA produced non-finite values"

import torchaudio  # noqa: E402
print("torchaudio", torchaudio.__version__)
print("SPIKE-GPU-OK")
EOF
