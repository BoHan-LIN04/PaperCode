#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/run_step.sh <command> <config> [extra args...]"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/multifuse/bin/python}"
COMMAND="$1"
CONFIG="$2"
shift 2

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

cd "$ROOT"
"$PYTHON_BIN" -m anthropic_emotions_repro.cli "$COMMAND" --config "$CONFIG" "$@"
