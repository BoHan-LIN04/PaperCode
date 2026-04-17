from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-0.6B"
    tokenizer_name_or_path: str | None = None
    device: str = "auto"
    trust_remote_code: bool = True
    torch_dtype: str | None = None


@dataclass
class DatasetConfig:
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"
    input_field: str = "input"
    target_field: str = "target"
    max_source_length: int = 256
    max_target_length: int = 128
    max_train_examples: int | None = None
    max_eval_examples: int | None = None


@dataclass
class PromptConfig:
    num_virtual_tokens: int = 20
    init_strategy: str = "random_uniform"
    prompt_path: str | None = None
    emotion_vector_route: str = "same_model"
    emotion_vectors_path: str | None = None
    emotion_vector_metadata_path: str | None = None
    emotion_vector_projection_path: str | None = None
    emotion_names: list[str] = field(default_factory=list)
    emotion_vector_combination: str = "repeat"
    emotion_vector_l2_normalize: bool = False
    random_range: float = 0.5
    sampled_vocab_size: int = 5000


@dataclass
class TrainingConfig:
    seed: int = 13
    batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    max_steps: int = 100
    eval_steps: int = 25
    save_steps: int = 25
    gradient_clip_norm: float = 1.0
    logging_steps: int = 10
    max_new_tokens: int = 64
    temperature: float = 1.0
    do_sample: bool = False
    top_k: int | None = None


@dataclass
class OutputConfig:
    output_dir: str = "artifacts/base"
    save_predictions: bool = True
    save_metrics: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    extends_value = raw.pop("extends", None)
    if extends_value is None:
        return raw
    parent_path = (path.parent / extends_value).resolve()
    parent = _load_yaml_with_extends(parent_path)
    return _deep_merge(parent, raw)


def _dict_to_dataclass(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        dataset=DatasetConfig(**data.get("dataset", {})),
        prompt=PromptConfig(**data.get("prompt", {})),
        training=TrainingConfig(**data.get("training", {})),
        output=OutputConfig(**data.get("output", {})),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    resolved = Path(path).resolve()
    raw = _load_yaml_with_extends(resolved)
    return _dict_to_dataclass(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "model": vars(config.model),
        "dataset": vars(config.dataset),
        "prompt": vars(config.prompt),
        "training": vars(config.training),
        "output": vars(config.output),
    }


def apply_override(config: ExperimentConfig, expression: str) -> ExperimentConfig:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got: {expression}")
    key_path, raw_value = expression.split("=", 1)
    parsed_value = yaml.safe_load(raw_value)
    target = config
    parts = key_path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], parsed_value)
    return config