from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "google/t5-v1_1-small"
    tokenizer_name_or_path: str | None = None
    device: str = "auto"


@dataclass
class DatasetConfig:
    task_name: str = "boolq"
    dataset_name: str = "super_glue"
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    max_source_length: int = 256
    max_target_length: int = 16


@dataclass
class PromptConfig:
    num_virtual_tokens: int = 20
    init_strategy: str = "class_labels"
    random_range: float = 0.5
    sampled_vocab_size: int = 5000


@dataclass
class TrainingConfig:
    seed: int = 13
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 0.3
    weight_decay: float = 1e-5
    parameter_scaling: bool = False
    warmup_steps: int = 0
    decay_factor: float | None = None
    steps_per_decay: int | None = None
    max_steps: int = 100
    eval_steps: int = 25
    save_steps: int = 25
    early_stopping_patience: int = 3
    generation_max_new_tokens: int = 16
    num_beams: int = 1
    gradient_clip_norm: float = 1.0
    logging_steps: int = 10


@dataclass
class AdaptationConfig:
    enabled: bool = False
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    prefix_fraction: float = 0.5
    max_source_length: int = 128
    max_target_length: int = 128
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "artifacts/lm-adapted"


@dataclass
class OutputConfig:
    output_dir: str = "artifacts/base"
    save_predictions: bool = True
    save_metrics: bool = True


@dataclass
class SweepConfig:
    enabled: bool = False
    parameter: str = "prompt.num_virtual_tokens"
    values: list[Any] = field(default_factory=list)


@dataclass
class BaselineConfig:
    mode: str = "prompt_tuning"
    multitask_tasks: list[str] = field(
        default_factory=lambda: ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]
    )
    add_task_prefix: bool = True


@dataclass
class CompareConfig:
    enabled: bool = False
    model_names: list[str] = field(
        default_factory=lambda: ["google/t5-v1_1-small", "google/t5-v1_1-base", "google/t5-v1_1-large"]
    )
    methods: list[str] = field(
        default_factory=lambda: ["prompt_tuning", "model_tuning", "model_tuning_multitask"]
    )
    seeds: list[int] = field(default_factory=lambda: [13, 17, 19])
    output_csv: str = "summary.csv"
    output_json: str = "runs.json"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)


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
        adaptation=AdaptationConfig(**data.get("adaptation", {})),
        output=OutputConfig(**data.get("output", {})),
        sweep=SweepConfig(**data.get("sweep", {})),
        baseline=BaselineConfig(**data.get("baseline", {})),
        compare=CompareConfig(**data.get("compare", {})),
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
        "adaptation": vars(config.adaptation),
        "output": vars(config.output),
        "sweep": vars(config.sweep),
        "baseline": vars(config.baseline),
        "compare": vars(config.compare),
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