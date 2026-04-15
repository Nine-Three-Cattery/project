#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 -m src.task2_retrieval "$@"
python3 -m src.validate_retrieval_qrels
python3 -m src.compute_p_at_5
python3 -m src.build_task2_analysis
