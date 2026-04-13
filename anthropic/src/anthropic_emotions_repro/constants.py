from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STEP_ORDER = [
    "00_env",
    "01_topic_bank",
    "02_prompt_templates",
    "03_synthetic_emotion_corpus",
    "04_activation_cache",
    "05_emotion_vectors",
    "report",
]

STEP_TO_DIR = {
    "prepare_topic_bank": "01_topic_bank",
    "prepare_prompt_templates": "02_prompt_templates",
    "generate_emotion_corpus": "03_synthetic_emotion_corpus",
    "extract_residuals": "04_activation_cache",
    "build_emotion_vectors": "05_emotion_vectors",
    "build_report": "report",
}
