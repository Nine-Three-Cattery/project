#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/detect_python.sh

ran_any=0

if [[ -n "${VERTEX_AI_API_KEY:-}" ]]; then
  "$PYTHON" -m src.task3_external_review --provider vertex --resume "$@"
  ran_any=1
else
  echo "Skipping Vertex audit because VERTEX_AI_API_KEY is not set."
fi

if [[ -n "${XAI_API_KEY:-}" ]]; then
  "$PYTHON" -m src.task3_external_review --provider xai --resume "$@"
  ran_any=1
else
  echo "Skipping xAI audit because XAI_API_KEY is not set."
fi

if [[ "$ran_any" -eq 0 ]]; then
  echo "No external audit was run because no provider API key was set." >&2
  exit 1
fi

"$PYTHON" -m src.build_task3_llm_audit
"$PYTHON" -m src.build_final_report
