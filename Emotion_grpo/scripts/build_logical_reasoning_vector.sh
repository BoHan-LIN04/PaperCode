#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/lbh/emorlenv/bin/python}"
DEVICE="${DEVICE:-cpu}"
SCORE_MODEL_NAME="${SCORE_MODEL_NAME:-Qwen/Qwen3-0.6B}"
LAYER_IDX="${LAYER_IDX:-8}"
TOKEN_POOL_START="${TOKEN_POOL_START:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-768}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/opt/data/private/lbh/PaperCode/Emotion_grpo/artifacts/logical_vectors/qwen3_0_6b_layer8}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export DEVICE SCORE_MODEL_NAME LAYER_IDX TOKEN_POOL_START BATCH_SIZE MAX_LENGTH ARTIFACT_DIR

"${PYTHON_BIN}" - <<'PY'
import os

from emotion_grpo.rewards.logical_vector_provider import build_logical_vector_artifact

outputs = build_logical_vector_artifact(
    source_files=[
        "/opt/data/private/lbh/PaperCode/Emotion_grpo/data/logical/AIME/AIME_results_Qwen3-32B_filtered.jsonl",
        "/opt/data/private/lbh/PaperCode/Emotion_grpo/data/logical/AMC/AMC_results_Qwen3-32B_filtered.jsonl",
        "/opt/data/private/lbh/PaperCode/Emotion_grpo/data/logical/Math/Math_results_Qwen3-32B_filtered.jsonl",
        "/opt/data/private/lbh/PaperCode/Emotion_grpo/data/logical/gsm8k/gsm8k_results_Qwen3-32B_filtered.jsonl",
    ],
    artifact_dir=os.environ["ARTIFACT_DIR"],
    score_model_name=os.environ["SCORE_MODEL_NAME"],
    layer_idx=int(os.environ["LAYER_IDX"]),
    token_pool_start=int(os.environ["TOKEN_POOL_START"]),
    batch_size=int(os.environ["BATCH_SIZE"]),
    text_fields=["thinking", "answer_content"],
    label_field="is_good",
    device=os.environ["DEVICE"],
    max_length=int(os.environ["MAX_LENGTH"]),
    use_neutral_pca=True,
    neutral_pca_variance=0.5,
)
for key, value in outputs.items():
    print(f"{key}={value}")
PY
