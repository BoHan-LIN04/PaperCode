from emotion_grpo.verl_reward import build_reward_inputs, compute_score


def test_reward_adapter_builds_messages_metadata_and_generation():
    batch_records, generations, metadata = build_reward_inputs(
        data_source="emotion_demo",
        solution_str="predicted emotion: joy",
        ground_truth="joy",
        extra_info={
            "messages": [{"role": "user", "content": "emotion?"}],
            "metadata": {"id": "demo-1", "score_hint": 0.75},
        },
    )

    assert generations == ["predicted emotion: joy"]
    assert batch_records[0]["messages"][0]["content"] == "emotion?"
    assert metadata[0]["score_hint"] == 0.75
    assert metadata[0]["ground_truth"] == "joy"


def test_reward_adapter_uses_configured_provider():
    result = compute_score(
        data_source="emotion_demo",
        solution_str="predicted emotion: joy",
        ground_truth="joy",
        extra_info={"messages": [], "metadata": {"score_hint": 0.75}},
        provider_cls="emotion_grpo.rewards.fixed_provider.FixedIntrinsicRewardProvider",
        provider_kwargs={"default_value": 0.1, "metadata_field": "score_hint"},
        include_details=True,
    )

    assert result["score"] == 0.75
    assert result["provider"] == "FixedIntrinsicRewardProvider"

