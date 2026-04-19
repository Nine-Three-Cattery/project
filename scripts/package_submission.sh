#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

ARCHIVE_PATH="$PROJECT_ROOT/submission.zip"

if [[ ! -d "$PROJECT_ROOT/submission" ]]; then
  echo "submission/ does not exist. Run bash scripts/prepare_submission.sh first." >&2
  exit 1
fi

rm -f "$ARCHIVE_PATH"
"$PYTHON" - "$PROJECT_ROOT/submission" "$ARCHIVE_PATH" <<'PY'
from pathlib import Path
import sys
import zipfile

submission_dir = Path(sys.argv[1])
archive_path = Path(sys.argv[2])

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(f"{submission_dir.name}/", "")
    for path in sorted(submission_dir.rglob("*")):
        relative_name = (Path(submission_dir.name) / path.relative_to(submission_dir)).as_posix()
        if path.is_dir():
            archive.writestr(f"{relative_name}/", "")
        else:
            archive.write(path, relative_name)
PY

echo "Created $ARCHIVE_PATH"
