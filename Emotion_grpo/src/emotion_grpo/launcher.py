from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_override_strings(config: dict[str, Any], override_strings: list[str]) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for override in override_strings:
        if "=" not in override:
            raise ValueError(f"Override must look like key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            lowered = raw_value.lower()
            if lowered == "true":
                value = True
            elif lowered == "false":
                value = False
            elif lowered == "null":
                value = None
            else:
                try:
                    value = int(raw_value)
                except ValueError:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
        _set_nested(merged, key, value)
    return merged


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        pairs: list[tuple[str, Any]] = []
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            pairs.extend(_flatten(nested_prefix, nested_value))
        return pairs
    return [(prefix, value)]


def _render_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def locate_reward_module() -> Path:
    from emotion_grpo import verl_reward

    return Path(verl_reward.__file__).resolve()


def build_verl_overrides(
    config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    reward_module_path: Path,
) -> list[str]:
    verl_cfg = copy.deepcopy(config["verl"])
    verl_cfg.setdefault("data", {})
    verl_cfg["data"]["train_files"] = str(train_file)
    verl_cfg["data"]["val_files"] = str(val_file)

    max_prompt_length = int(verl_cfg["data"]["max_prompt_length"])
    max_response_length = int(verl_cfg["data"]["max_response_length"])
    rollout_cfg = verl_cfg.setdefault("actor_rollout_ref", {}).setdefault("rollout", {})
    rollout_cfg.setdefault("max_model_len", max_prompt_length + max_response_length)

    reward_cfg = verl_cfg.setdefault("reward", {})
    reward_cfg["custom_reward_function"] = {
        "path": str(reward_module_path),
        "name": "compute_score",
        "reward_kwargs": {
            "provider_cls": config["reward"]["provider_cls"],
            "provider_kwargs": config["reward"].get("provider_kwargs", {}),
            "include_details": config["reward"].get("include_details", True),
        },
    }

    pairs = _flatten("", verl_cfg)
    return [f"{key}={_render_override_value(value)}" for key, value in pairs]


def build_training_command(
    config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    working_config = apply_override_strings(config, extra_overrides or [])
    reward_module_path = locate_reward_module()
    overrides = build_verl_overrides(
        config=working_config,
        train_file=train_file,
        val_file=val_file,
        reward_module_path=reward_module_path,
    )
    python_bin = working_config["project"]["python_bin"]
    return [python_bin, "-m", "verl.trainer.main_ppo", *overrides]


def run_training_command(command: list[str]) -> None:
    subprocess.run(command, check=True)

