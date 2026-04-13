from __future__ import annotations

import argparse
from pathlib import Path

from emotion_grpo.config import load_experiment_config
from emotion_grpo.data import ensure_demo_parquet
from emotion_grpo.launcher import build_training_command, run_training_command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch VERL GRPO training with an intrinsic reward provider.")
    parser.add_argument("--config-name", required=True, help="Experiment config under configs/experiment/")
    parser.add_argument("--config-dir", default=None, help="Optional config directory override")
    parser.add_argument("--dry-run", action="store_true", help="Print the final VERL command and exit")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare parquet files and exit")
    parser.add_argument("--overwrite-data", action="store_true", help="Regenerate parquet files from JSONL")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted paths, for example --set reward.provider_kwargs.seed=11",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_experiment_config(config_name=args.config_name, config_dir=args.config_dir)
    train_file, val_file = ensure_demo_parquet(config=config, overwrite=args.overwrite_data)

    print(f"[emotion_grpo] config={args.config_name}")
    print(f"[emotion_grpo] train_parquet={train_file}")
    print(f"[emotion_grpo] val_parquet={val_file}")

    if args.prepare_only:
        return

    command = build_training_command(
        config=config,
        train_file=Path(train_file),
        val_file=Path(val_file),
        extra_overrides=args.set,
    )
    print("[emotion_grpo] command=")
    print(" ".join(command))

    if args.dry_run:
        return

    run_training_command(command)


if __name__ == "__main__":
    main()

