#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/lbh/emorlenv/bin/python}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m emotion_grpo.cli.train --config-name emotion_vector_qwen3_0_6b_demo "$@"
