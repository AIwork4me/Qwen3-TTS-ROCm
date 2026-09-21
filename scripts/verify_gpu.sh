#!/usr/bin/env bash
# verify_gpu.sh — quick ROCm/GPU sanity check for the Qwen3-TTS-ROCm project.
# Validates that the AMD pip-wheel torch stack sees the gfx1151 GPU and can
# run bf16 matmul + SDPA + torchaudio import. Exits non-zero on any failure.
#
# Usage: bash scripts/verify_gpu.sh   (run from repo root, expects .venv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README.md Quick Start)." >&2
    exit 1
fi

"$PY" - <<'EOF'
import sys
import torch
import torch.nn.functional as F

print("torch", torch.__version__, "| HIP", torch.version.hip)
assert torch.version.hip is not None, "not a ROCm build of torch"
assert torch.version.hip.startswith("7.14"), (
    f"expected HIP 7.14.x, got {torch.version.hip}"
)
assert torch.cuda.is_available(), "cuda not available under ROCm build"

dev = "cuda:0"
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name, "| arch:", getattr(props, "gcnArchName", "n/a"))
assert getattr(props, "gcnArchName", "") == "gfx1151", "unexpected GPU arch"

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
