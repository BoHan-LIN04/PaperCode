from emotion_grpo.rewards.logical_exact_match_provider import (
    LogicalExactMatchRewardProvider,
    normalize_logical_answer,
)


def test_normalize_logical_answer_extracts_boxed_and_numbers():
    assert normalize_logical_answer(r"The final answer is \boxed{42}.") == "42"
    assert normalize_logical_answer("Answer: 003") == "003"


def test_logical_exact_match_provider_scores_exact_match():
    provider = LogicalExactMatchRewardProvider(correct_reward=1.0, incorrect_reward=-0.5)
    scores = provider.score_batch(
        batch_records=[{}],
        generations=["The final answer is 42."],
        metadata=[{"ground_truth": "42"}],
    )
    assert scores == [1.0]


def test_logical_exact_match_provider_scores_incorrect_match():
    provider = LogicalExactMatchRewardProvider(correct_reward=1.0, incorrect_reward=-0.5)
    scores = provider.score_batch(
        batch_records=[{}],
        generations=["17"],
        metadata=[{"ground_truth": "42"}],
    )
    assert scores == [-0.5]
