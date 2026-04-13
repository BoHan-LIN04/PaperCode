#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/lbh/emorlenv/bin/python}"
VERL_COMMIT="1a771692d441ec249bf434060fe6c9859ab28e19"
VERL_LOCAL_SRC="${VERL_LOCAL_SRC:-/tmp/verl_v071_inspect}"
PIP_INDEX_URL_DEFAULT="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_FALLBACK_INDEX_URL="${PIP_FALLBACK_INDEX_URL:-https://pypi.org/simple}"
CODETIMING_SHIM_DIR="${PROJECT_ROOT}/vendor/codetiming_shim"
DISABLE_LOCAL_PROXY_FOR_PIP="${DISABLE_LOCAL_PROXY_FOR_PIP:-0}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="${PIP_INDEX_URL_DEFAULT}"
fi

if [[ -z "${PIP_TRUSTED_HOST:-}" ]]; then
  export PIP_TRUSTED_HOST="$("${PYTHON_BIN}" - <<'PY'
from urllib.parse import urlparse
import os
print(urlparse(os.environ["PIP_INDEX_URL"]).hostname or "")
PY
)"
fi

detect_loopback_proxy() {
  "${PYTHON_BIN}" - <<'PY'
from urllib.parse import urlparse
import os

proxy_hosts = []
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    value = os.environ.get(key)
    if value:
        proxy_hosts.append((urlparse(value).hostname or "").lower())

has_loopback = any(host in {"127.0.0.1", "localhost"} for host in proxy_hosts)
print("1" if has_loopback else "0")
PY
}

if [[ "${DISABLE_LOCAL_PROXY_FOR_PIP}" == "0" ]] && [[ "$(detect_loopback_proxy)" == "1" ]]; then
  DISABLE_LOCAL_PROXY_FOR_PIP="1"
fi

echo "Using python: ${PYTHON_BIN}"
echo "Using pip index: ${PIP_INDEX_URL}"
echo "Fallback pip index: ${PIP_FALLBACK_INDEX_URL}"
echo "Disable local proxy for pip: ${DISABLE_LOCAL_PROXY_FOR_PIP}"

run_pip() {
  if [[ "${DISABLE_LOCAL_PROXY_FOR_PIP}" == "1" ]]; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY "${PYTHON_BIN}" -m pip "$@"
  else
    "${PYTHON_BIN}" -m pip "$@"
  fi
}

run_pip_with_fallback() {
  local primary_index="${PIP_INDEX_URL}"
  local fallback_index="${PIP_FALLBACK_INDEX_URL}"

  if run_pip install --no-build-isolation "$@"; then
    return 0
  fi

  echo "Primary install failed. Retrying without local proxy against ${fallback_index}."
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    "${PYTHON_BIN}" -m pip install --no-build-isolation -i "${fallback_index}" "$@"
}

run_pip_with_fallback -e "${PROJECT_ROOT}[dev]"
run_pip_with_fallback pandas pyarrow
run_pip_with_fallback hydra-core omegaconf
run_pip_with_fallback -e "${CODETIMING_SHIM_DIR}"

if [[ -d "${VERL_LOCAL_SRC}/.git" ]]; then
  LOCAL_COMMIT="$(git -C "${VERL_LOCAL_SRC}" rev-parse HEAD)"
  if [[ "${LOCAL_COMMIT}" == "${VERL_COMMIT}" ]]; then
    echo "Installing VERL from local checkout: ${VERL_LOCAL_SRC}"
    run_pip_with_fallback "${VERL_LOCAL_SRC}"
  else
    echo "Local VERL checkout commit ${LOCAL_COMMIT} does not match ${VERL_COMMIT}, falling back to GitHub."
    run_pip_with_fallback "git+https://github.com/volcengine/verl.git@${VERL_COMMIT}"
  fi
else
  run_pip_with_fallback "git+https://github.com/volcengine/verl.git@${VERL_COMMIT}"
fi

echo "Installed Emotion_grpo and VERL ${VERL_COMMIT}"
