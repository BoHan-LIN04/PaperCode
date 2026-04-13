#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[tree_report] project root: $ROOT"
find "$ROOT" -maxdepth 3 | sed -n '1,400p'
