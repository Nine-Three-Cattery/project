#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

$PYTHON -m src.task2_retrieval "$@"
$PYTHON -m src.validate_retrieval_qrels
$PYTHON -m src.compute_p_at_5
$PYTHON -m src.build_task2_analysis
