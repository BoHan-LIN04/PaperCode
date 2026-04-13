from emotion_grpo.rewards.random_provider import RandomIntrinsicRewardProvider


def test_random_provider_is_reproducible():
    provider_a = RandomIntrinsicRewardProvider(seed=13, min_value=-0.5, max_value=0.5)
    provider_b = RandomIntrinsicRewardProvider(seed=13, min_value=-0.5, max_value=0.5)

    batch_records = [{"messages": [{"role": "user", "content": "hello"}], "metadata": {"id": "r1"}}]
    generations = ["world"]
    metadata = [{"id": "r1"}]

    scores_a = provider_a.score_batch(batch_records, generations, metadata)
    scores_b = provider_b.score_batch(batch_records, generations, metadata)

    assert scores_a == scores_b
    assert len(scores_a) == 1
    assert -0.5 <= scores_a[0] <= 0.5

