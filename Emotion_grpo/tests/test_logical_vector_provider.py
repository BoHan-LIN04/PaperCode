from pathlib import Path

from emotion_grpo.rewards.logical_vector_provider import (
    _score_against_logic_centroids,
    load_reasoning_examples,
)


def test_load_reasoning_examples_reads_thinking_and_labels(tmp_path: Path):
    source = tmp_path / "results.jsonl"
    source.write_text(
        '{"thinking":"good path","answer_content":"good answer","is_good":true}\n'
        '{"thinking":"bad path","answer_content":"bad answer","is_good":false}\n',
        encoding="utf-8",
    )
    texts, labels = load_reasoning_examples([source])
    assert texts == ["good path", "bad path"]
    assert labels == [True, False]


def test_logical_vector_margin_score_prefers_positive_centroid():
    good = [1.0, 0.0]
    bad = [0.0, 1.0]
    score = _score_against_logic_centroids(
        embedding=good,
        positive_centroid=good,
        negative_centroid=bad,
        score_mode="margin",
        reward_scale=1.0,
        reward_clip=1.0,
    )
    assert score > 0.9
