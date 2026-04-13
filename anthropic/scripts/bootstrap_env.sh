#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/multifuse/bin/python}"

echo "[bootstrap] root=$ROOT"
echo "[bootstrap] python=$PYTHON_BIN"

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -e "$ROOT[dev]"

echo "[bootstrap] done"
