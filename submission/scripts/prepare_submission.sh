#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SUBMISSION_DIR="$PROJECT_ROOT/submission"

copy_path() {
  local source_path="$1"
  local destination_dir="$2"
  if [[ -e "$source_path" ]]; then
    mkdir -p "$destination_dir"
    cp -R "$source_path" "$destination_dir/"
  fi
}

find "$PROJECT_ROOT" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name ".DS_Store" -delete

bash scripts/run_full_pipeline.sh
find "$PROJECT_ROOT" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name ".DS_Store" -delete
python3 -B -m src.verify_submission --root "$PROJECT_ROOT"

rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

copy_path README.md "$SUBMISSION_DIR"
copy_path preprocess.py "$SUBMISSION_DIR"
copy_path requirements.txt "$SUBMISSION_DIR"
copy_path requirements-lock.txt "$SUBMISSION_DIR"
copy_path .python-version "$SUBMISSION_DIR"
copy_path configs "$SUBMISSION_DIR"
copy_path data "$SUBMISSION_DIR"
copy_path docs "$SUBMISSION_DIR"
copy_path outputs "$SUBMISSION_DIR"
copy_path scripts "$SUBMISSION_DIR"
copy_path src "$SUBMISSION_DIR"

find "$SUBMISSION_DIR" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find "$SUBMISSION_DIR" -name ".DS_Store" -delete

python3 -B -m src.verify_submission --root "$SUBMISSION_DIR"

echo "Prepared a clean submission bundle at $SUBMISSION_DIR."
