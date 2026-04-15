#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARCHIVE_PATH="$PROJECT_ROOT/submission.zip"

if [[ ! -d "$PROJECT_ROOT/submission" ]]; then
  echo "submission/ does not exist. Run bash scripts/prepare_submission.sh first." >&2
  exit 1
fi

rm -f "$ARCHIVE_PATH"
ditto -c -k --sequesterRsrc --keepParent "$PROJECT_ROOT/submission" "$ARCHIVE_PATH"

echo "Created $ARCHIVE_PATH"
