#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/run_pipeline.sh <config> [extra args...]"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
shift 1

STEPS=(
  "prepare_topic_bank"
  "prepare_prompt_templates"
  "generate_emotion_corpus"
  "extract_residuals"
  "build_emotion_vectors"
  "build_report"
)

TOTAL="${#STEPS[@]}"

render_bar() {
  local current="$1"
  local total="$2"
  local width=30
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  printf '['
  printf '%0.s#' $(seq 1 "$filled")
  printf '%0.s-' $(seq 1 "$empty")
  printf ']'
}

cd "$ROOT"

for idx in "${!STEPS[@]}"; do
  step="${STEPS[$idx]}"
  current=$(( idx + 1 ))
  bar="$(render_bar "$current" "$TOTAL")"
  echo "[pipeline] ${bar} ${current}/${TOTAL} -> ${step}"
  bash scripts/run_step.sh "$step" "$CONFIG" "$@"
done

echo "[pipeline] complete"
