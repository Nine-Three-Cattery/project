#!/usr/bin/env bash
# Detect a suitable Python >= 3.10 interpreter.
#
# Usage: source this file from any project script.
#   source "$(dirname "${BASH_SOURCE[0]}")/detect_python.sh"
# After sourcing, use $PYTHON instead of python3.

if [[ -n "${PYTHON:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_check_python_version() {
  local py="$1"
  "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null
}

# Priority: venv > VIRTUAL_ENV > homebrew > system
for candidate in \
  "${PROJECT_ROOT:-.}/.venv/bin/python3" \
  "${VIRTUAL_ENV:-__none__}/bin/python3" \
  /opt/homebrew/bin/python3.14 \
  /opt/homebrew/bin/python3.13 \
  /opt/homebrew/bin/python3.12 \
  /opt/homebrew/bin/python3.11 \
  /opt/homebrew/bin/python3.10 \
  /opt/homebrew/bin/python3 \
  /usr/local/bin/python3 \
  python3; do
  if [[ "$candidate" == *"__none__"* ]]; then
    continue
  fi
  if command -v "$candidate" &>/dev/null && _check_python_version "$candidate"; then
    export PYTHON="$candidate"
    break
  fi
done

if [[ -z "${PYTHON:-}" ]]; then
  echo "ERROR: No Python >= 3.10 found. Install Python 3.10+ or set the PYTHON environment variable." >&2
  exit 1
fi
