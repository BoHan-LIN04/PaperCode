from __future__ import annotations

import argparse
import json

from .config import apply_override, load_experiment_config
from .training import evaluate_prompt_model, train_prompt_model, validate_experiment_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decoder-only soft prompt tuning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["train", "eval", "validate-config"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--override", action="append", default=[])
        if name == "eval":
            sub.add_argument("--prompt-path", required=True)

    return parser


def _load_config_with_overrides(config_path: str, overrides: list[str]):
    config = load_experiment_config(config_path)
    for override in overrides:
        apply_override(config, override)
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_config_with_overrides(args.config, args.override)

    if args.command == "train":
        result = train_prompt_model(config)
    elif args.command == "eval":
        result = evaluate_prompt_model(config, prompt_path=args.prompt_path)
    elif args.command == "validate-config":
        result = validate_experiment_config(config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()