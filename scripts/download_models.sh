#!/usr/bin/env bash
# download_models.sh — fetch official Qwen3-TTS repositories via the project
# downloader (ModelScope-first with hf-mirror fallback; already-downloaded
# repos short-circuit so re-runs are cheap).
#
# Usage:
#   bash scripts/download_models.sh [alias ...]     # no args = all six aliases
#
# Aliases:
#   tokenizer  voice-design  custom-voice  base  custom-voice-0.6b  base-0.6b
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found — run scripts/install.sh first." >&2
    exit 1
fi

ALL_ALIASES=(tokenizer voice-design custom-voice base custom-voice-0.6b base-0.6b)
REQUESTED=("$@")
if (( ${#REQUESTED[@]} == 0 )); then
    REQUESTED=("${ALL_ALIASES[@]}")
fi

MODELS_ROOT="$($PY -c 'from qwen3_tts_rocm.models import local_dir; print(local_dir())')"
echo "==> downloading ${#REQUESTED[@]} repo(s): ${REQUESTED[*]}"
echo "==> target root: $MODELS_ROOT"

# download([...]) validates aliases itself (KeyError -> non-zero exit) and
# prints nothing; per-repo progress bars come from ModelScope/huggingface_hub.
DIRS="$("$PY" - "${REQUESTED[@]}" <<'EOF'
import sys
from qwen3_tts_rocm.models import download

for path in download(sys.argv[1:]):
    print(path)
EOF
)"

echo "==> fetched/skipped repos on disk:"
while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    printf '    %-38s %s\n' "$(basename "$dir")" "$(du -sh "$dir" | cut -f1)"
done <<< "$DIRS"

echo "MODEL-DOWNLOAD-OK"
