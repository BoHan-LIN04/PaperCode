from __future__ import annotations

import argparse
import json

from .config import apply_override, load_experiment_config, DatasetConfig
from .training import (
    adapt_language_model,
    ensemble_prompt_models,
    evaluate_prompt_model,
    render_figure1_plot,
    run_model_comparison,
    run_sweep,
    train_model_tuning,
    train_model_tuning_multitask,
    train_prompt_model,
)
from .reporting import generate_comparison_report
from .interpretability import analyze_prompt_interpretability, extract_nearest_tokens_for_display
from .dataset_validation import (
    validate_label_distribution,
    compute_label_distribution,
    print_label_distribution,
    validate_all_superglue,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soft prompt tuning reproduction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["train", "eval", "sweep", "adapt-lm", "train-model", "train-multitask", "compare"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--override", action="append", default=[])
        if name == "eval":
            sub.add_argument("--prompt-path", required=True)

    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--config", required=True)
    ensemble.add_argument("--override", action="append", default=[])
    ensemble.add_argument("--prompt", dest="prompts", action="append", required=True)

    analyze = subparsers.add_parser("analyze-prompt")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--prompt-path", required=True)
    analyze.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    analyze.add_argument("--output", default=None, help="Optional output file for analysis results")

    plot = subparsers.add_parser("plot-figure1")
    plot.add_argument("--summary-csv", required=True)
    plot.add_argument("--output", required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--summary-csv", required=True)
    report.add_argument("--figure", default=None)
    report.add_argument("--output", default=None)

    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument(
        "--task",
        default=None,
        help="Specific task to validate (e.g., boolq). If not provided, validates all SuperGLUE tasks.",
    )
    validate.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Data split to validate",
    )
    validate.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Tolerance in percentage points for label distribution validation",
    )

    return parser


def _load_config_with_overrides(config_path: str, overrides: list[str]):
    config = load_experiment_config(config_path)
    for override in overrides:
        apply_override(config, override)
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "plot-figure1":
        result = render_figure1_plot(args.summary_csv, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "report":
        result = generate_comparison_report(args.summary_csv, args.figure, args.output)
        print(json.dumps({"report_path": result}, indent=2, ensure_ascii=False))
        return

    if args.command == "validate-dataset":
        if args.task is None:
            # Validate all SuperGLUE tasks
            validate_all_superglue()
        else:
            # Validate single task
            dataset_name = "glue" if args.task in ["mrpc", "qqp"] else "super_glue"
            config = DatasetConfig(task_name=args.task, dataset_name=dataset_name)
            print(f"\nValidating {args.task} ({args.split} split)...")
            dist = compute_label_distribution(config, args.split)
            print_label_distribution(dist)
            print("\nComparing with paper values:")
            validate_label_distribution(config, args.split, tolerance=args.tolerance)
        return

    if args.command == "analyze-prompt":
        config = _load_config_with_overrides(args.config, [])
        result = analyze_prompt_interpretability(
            model_name_or_path=config.model.name_or_path,
            prompt_path=args.prompt_path,
            num_virtual_tokens=config.prompt.num_virtual_tokens,
            init_strategy=config.prompt.init_strategy,
            random_range=config.prompt.random_range,
            sampled_vocab_size=config.prompt.sampled_vocab_size,
            label_texts=None,
            k=args.k,
            device=config.model.device,
        )
        # Print human-readable analysis
        print(extract_nearest_tokens_for_display(result))
        # Also output JSON
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return

    config = _load_config_with_overrides(args.config, args.override)

    if args.command == "train":
        result = train_prompt_model(config)
    elif args.command == "train-model":
        result = train_model_tuning(config)
    elif args.command == "train-multitask":
        result = train_model_tuning_multitask(config)
    elif args.command == "eval":
        result = evaluate_prompt_model(config, prompt_path=args.prompt_path)
    elif args.command == "sweep":
        result = run_sweep(config)
    elif args.command == "ensemble":
        result = ensemble_prompt_models(config, prompt_paths=args.prompts)
    elif args.command == "adapt-lm":
        result = adapt_language_model(config)
    elif args.command == "compare":
        result = run_model_comparison(config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
