#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

bash scripts/run_task1.sh
bash scripts/run_retrieval.sh
bash scripts/run_task3.sh
$PYTHON -m src.build_final_report
