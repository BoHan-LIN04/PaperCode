from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("__replace__") is True:
            replacement = {nested_key: copy.deepcopy(nested_value) for nested_key, nested_value in value.items() if nested_key != "__replace__"}
            merged[key] = replacement
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file {path}")
    return data


def _resolve_path(path_value: str | None, project_root: Path) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((project_root / candidate).resolve())


def load_experiment_config(config_name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    resolved_config_dir = Path(config_dir) if config_dir is not None else project_root / "configs"
    experiment_path = resolved_config_dir / "experiment" / f"{config_name}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    experiment_cfg = _load_yaml(experiment_path)
    defaults = experiment_cfg.pop("defaults", {})

    base_name = defaults.get("base", "base")
    model_name = defaults.get("model")
    runtime_name = defaults.get("runtime")

    merged = _load_yaml(resolved_config_dir / f"{base_name}.yaml")
    if model_name:
        merged = _deep_merge(merged, _load_yaml(resolved_config_dir / "model" / f"{model_name}.yaml"))
    if runtime_name:
        merged = _deep_merge(merged, _load_yaml(resolved_config_dir / "runtime" / f"{runtime_name}.yaml"))
    merged = _deep_merge(merged, experiment_cfg)

    merged.setdefault("project", {})
    merged["project"]["root"] = str(project_root)
    merged["project"]["workspace"] = _resolve_path(merged["project"].get("workspace", "."), project_root)
    merged["project"]["python_bin"] = _resolve_path(merged["project"].get("python_bin"), project_root)

    merged.setdefault("datasets", {})
    for key in ("train_jsonl", "val_jsonl", "processed_train", "processed_val"):
        merged["datasets"][key] = _resolve_path(merged["datasets"].get(key), project_root)

    trainer_cfg = merged.setdefault("verl", {}).setdefault("trainer", {})
    if "default_local_dir" in trainer_cfg:
        trainer_cfg["default_local_dir"] = _resolve_path(trainer_cfg["default_local_dir"], project_root)

    merged["_meta"] = {
        "config_name": config_name,
        "config_dir": str(resolved_config_dir),
        "project_root": str(project_root),
    }
    return merged
