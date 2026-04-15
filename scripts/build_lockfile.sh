#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

$PYTHON -m venv "$TEMP_DIR/venv"
source "$TEMP_DIR/venv/bin/activate"

$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt
$PYTHON -m pip freeze --exclude-editable | LC_ALL=C sort > requirements-lock.txt
$PYTHON - <<'PY'
import platform
from pathlib import Path

Path(".python-version").write_text(platform.python_version() + "\n", encoding="utf-8")
PY

echo "Wrote requirements-lock.txt from a clean temporary virtual environment."
echo "Wrote .python-version for the current interpreter."
