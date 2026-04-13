import numpy as np

from emotion_grpo.rewards.emotion_vector_provider import (
    canonicalize_emotion_name,
    compute_neutral_components,
    parse_layer_idx_from_artifact_root,
    project_out_components,
    score_embedding_against_vectors,
)


def test_parse_layer_idx_from_artifact_root():
    path = "/tmp/model=qwen3-0.6b__lang=en__emotions=12__topics=8__stories=2__layer=8"
    assert parse_layer_idx_from_artifact_root(path) == 8


def test_canonicalize_emotion_name_uses_aliases():
    aliases = {"joy": "joyful"}
    assert canonicalize_emotion_name(" Joy ", aliases=aliases) == "joyful"


def test_project_out_components_removes_axis():
    embedding = np.asarray([3.0, 4.0], dtype=np.float32)
    components = np.asarray([[1.0, 0.0]], dtype=np.float32)
    projected = project_out_components(embedding, components)
    assert np.allclose(projected, np.asarray([0.0, 4.0], dtype=np.float32))


def test_compute_neutral_components_uses_requested_pc_count():
    neutral = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    components = compute_neutral_components(neutral, pc_count=1)
    assert components.shape == (1, 2)
    assert np.allclose(np.abs(components[0]), np.asarray([1.0, 0.0], dtype=np.float32))


def test_score_embedding_against_vectors_margin_prefers_target_over_others():
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    score = score_embedding_against_vectors(
        embedding,
        vectors,
        target_index=0,
        score_mode="margin",
        reward_scale=1.0,
        reward_clip=1.0,
    )
    assert score > 0.9
