from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decoder_soft_prompt_repro.config import PromptConfig
from decoder_soft_prompt_repro.training import _is_better_checkpoint, validate_prompt_configuration


def test_validate_prompt_configuration_same_model_accepts_matching_sizes(tmp_path: Path):
    vectors = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    vectors_path = tmp_path / "vectors.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(vectors_path, vectors)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful"]}), encoding="utf-8")

    prompt = PromptConfig(
        init_strategy="emotion_vectors",
        emotion_vector_route="same_model",
        emotion_vectors_path=str(vectors_path),
        emotion_vector_metadata_path=str(metadata_path),
        emotion_names=["joyful"],
    )

    result = validate_prompt_configuration(prompt, model_hidden_size=3)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_prompt_configuration_projected_requires_projection(tmp_path: Path):
    vectors = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    vectors_path = tmp_path / "vectors.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(vectors_path, vectors)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful"]}), encoding="utf-8")

    prompt = PromptConfig(
        init_strategy="emotion_vectors",
        emotion_vector_route="projected",
        emotion_vectors_path=str(vectors_path),
        emotion_vector_metadata_path=str(metadata_path),
        emotion_names=["joyful"],
    )

    result = validate_prompt_configuration(prompt, model_hidden_size=4)
    assert result["valid"] is False
    assert any("projection_path is required" in message for message in result["errors"])


def test_validate_prompt_configuration_same_model_rejects_projection(tmp_path: Path):
    vectors = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    projection = np.eye(3, dtype=np.float32)
    vectors_path = tmp_path / "vectors.npy"
    metadata_path = tmp_path / "metadata.json"
    projection_path = tmp_path / "projection.npy"
    np.save(vectors_path, vectors)
    np.save(projection_path, projection)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful"]}), encoding="utf-8")

    prompt = PromptConfig(
        init_strategy="emotion_vectors",
        emotion_vector_route="same_model",
        emotion_vectors_path=str(vectors_path),
        emotion_vector_metadata_path=str(metadata_path),
        emotion_vector_projection_path=str(projection_path),
        emotion_names=["joyful"],
    )

    result = validate_prompt_configuration(prompt, model_hidden_size=3)
    assert result["valid"] is False
    assert any("must be empty" in message for message in result["errors"])


def test_best_checkpoint_prefers_lower_eval_loss_when_exact_match_ties():
    assert _is_better_checkpoint(0.0, 2.2, 0.0, 2.4) is True
    assert _is_better_checkpoint(0.0, 2.4, 0.0, 2.2) is False


def test_best_checkpoint_prefers_higher_exact_match_over_lower_loss():
    assert _is_better_checkpoint(0.1, 2.9, 0.0, 2.1) is True
    assert _is_better_checkpoint(0.0, 1.9, 0.1, 3.0) is False