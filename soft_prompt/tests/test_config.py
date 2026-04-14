from pathlib import Path

from soft_prompt_repro.config import apply_override, load_experiment_config


def test_load_config_with_extends():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ablations" / "prompt_length.yaml"
    config = load_experiment_config(config_path)

    assert config.sweep.enabled is True
    assert config.prompt.num_virtual_tokens == 20
    assert config.dataset.task_name == "boolq"


def test_apply_override_updates_nested_value():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_experiment_config(config_path)
    apply_override(config, "prompt.num_virtual_tokens=7")
    apply_override(config, "dataset.task_name=rte")

    assert config.prompt.num_virtual_tokens == 7
    assert config.dataset.task_name == "rte"