from pathlib import Path

from emotion_grpo.config import load_experiment_config
from emotion_grpo.launcher import build_verl_overrides


def test_build_verl_overrides_includes_custom_reward_hook():
    config = load_experiment_config("single_gpu_demo")
    overrides = build_verl_overrides(
        config=config,
        train_file=Path("/tmp/train.parquet"),
        val_file=Path("/tmp/val.parquet"),
        reward_module_path=Path("/tmp/verl_reward.py"),
    )

    override_text = "\n".join(overrides)
    assert 'algorithm.adv_estimator="grpo"' in override_text
    assert 'reward.custom_reward_function.path="/tmp/verl_reward.py"' in override_text
    assert 'reward.custom_reward_function.name="compute_score"' in override_text
    assert (
        'reward.custom_reward_function.reward_kwargs.provider_cls='
        '"emotion_grpo.rewards.random_provider.RandomIntrinsicRewardProvider"' in override_text
    )
    assert 'actor_rollout_ref.model.path="Qwen/Qwen2.5-0.5B-Instruct"' in override_text


def test_smoke_test_replaces_provider_kwargs_when_switching_provider():
    config = load_experiment_config("smoke_test")

    assert config["reward"]["provider_cls"] == "emotion_grpo.rewards.fixed_provider.FixedIntrinsicRewardProvider"
    assert config["reward"]["provider_kwargs"] == {"default_value": 0.25}
