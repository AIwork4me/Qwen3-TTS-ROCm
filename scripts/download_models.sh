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

# --- -h/--help: static usage, works even before scripts/install.sh ----------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
download_models.sh — download the official Qwen3-TTS model repositories
(ModelScope-first with hf-mirror fallback; already-downloaded repos are skipped).

Usage:
  bash scripts/download_models.sh [alias ...]     # no args = all six aliases
  bash scripts/download_models.sh --help

Aliases (approx. download size):
  tokenizer           651M    audio codec — required by every demo/pipeline
  custom-voice        4.3G    reference-voice cloning (1.7B)
  voice-design        4.3G    text-described voice creation (1.7B)
  base                4.3G    base TTS (1.7B)
  custom-voice-0.6b   2.4G    smaller custom-voice
  base-0.6b           2.4G    smaller base

Environment:
  QWEN3_TTS_ROCM_MODELS_DIR   override the models root (default: <repo>/models)

Examples:
  bash scripts/download_models.sh                         # everything (~18GB)
  bash scripts/download_models.sh tokenizer custom-voice  # minimal demo subset
  bash scripts/download_models.sh tokenizer base-0.6b     # smaller 0.6B subset
USAGE
    exit 0
fi

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

# download([...]) does the fetching; per-repo progress bars come from
# ModelScope/huggingface_hub.  UX-fix U2: an unknown alias must NEVER escape
# as a Python traceback — validate first and catch KeyError right here,
# printing one clean line (with the known aliases) and exiting 2.
DIRS="$("$PY" - "${REQUESTED[@]}" <<'EOF'
import sys

from qwen3_tts_rocm.models import ALIASES, REPOS, download

requested = sys.argv[1:]
bad = next((a for a in requested if a not in REPOS), None)
if bad is not None:
    print(
        f"ERROR: unknown model alias {bad!r} (未知的模型别名) — "
        f"known aliases: {', '.join(ALIASES)}",
        file=sys.stderr,
    )
    sys.exit(2)
try:
    for path in download(requested):
        print(path)
except KeyError as exc:  # defensive: registry-level rejection stays one line
    print(f"ERROR: {exc.args[0] if exc.args else exc}", file=sys.stderr)
    sys.exit(2)
EOF
)"

echo "==> fetched/skipped repos on disk:"
while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    printf '    %-38s %s\n' "$(basename "$dir")" "$(du -sh "$dir" | cut -f1)"
done <<< "$DIRS"

echo "MODEL-DOWNLOAD-OK"
