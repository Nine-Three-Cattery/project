#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

LOCK_VENV="$TEMP_DIR/venv"
"$PYTHON" -m venv "$LOCK_VENV"

LOCK_PYTHON=""
for candidate in \
  "$LOCK_VENV/bin/python3" \
  "$LOCK_VENV/bin/python" \
  "$LOCK_VENV/Scripts/python.exe" \
  "$LOCK_VENV/Scripts/python"; do
  if [[ -x "$candidate" ]] && _check_python_version "$candidate"; then
    LOCK_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$LOCK_PYTHON" ]]; then
  echo "ERROR: Could not find Python >= 3.10 inside temporary virtual environment." >&2
  exit 1
fi

"$LOCK_PYTHON" -m pip install --upgrade pip
"$LOCK_PYTHON" -m pip install -r requirements.txt
"$LOCK_PYTHON" -m pip freeze --exclude-editable | LC_ALL=C sort > requirements-lock.txt
"$LOCK_PYTHON" - <<'PY'
import platform
from pathlib import Path

Path(".python-version").write_text(platform.python_version() + "\n", encoding="utf-8")
PY

echo "Wrote requirements-lock.txt from a clean temporary virtual environment."
echo "Wrote .python-version for the current interpreter."
