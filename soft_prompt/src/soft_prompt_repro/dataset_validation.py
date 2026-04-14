"""
Data validation module for verifying dataset label distributions.

This module implements validation checks against the paper's reported label distributions
(Appendix A.3, Tables 8-16) to ensure correct data loading and preprocessing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from .config import DatasetConfig
from .tasks import get_task_spec


@dataclass
class LabelDistribution:
    """Label distribution statistics for a dataset."""

    task_name: str
    split: str
    total_examples: int
    label_counts: dict[str, int]
    label_percentages: dict[str, float]


# Paper's reported label distributions (Tables 8-16)
PAPER_LABEL_DISTRIBUTIONS = {
    "boolq": {
        "train": {"False": 37.7, "True": 62.3},
        "validation": {"False": 37.8, "True": 62.2},
    },
    "cb": {
        "train": {"contradiction": 47.6, "entailment": 46.0, "neutral": 6.4},
        "validation": {"contradiction": 50.0, "entailment": 41.1, "neutral": 8.9},
    },
    "copa": {
        "train": {"choice1": 48.8, "choice2": 51.2},
        "validation": {"choice1": 55.0, "choice2": 45.0},
    },
    "multirc": {
        "train": {"False": 55.9, "True": 44.1},
        "validation": {"False": 57.2, "True": 42.8},
    },
    "wic": {
        "train": {"False": 50.0, "True": 50.0},
        "validation": {"False": 50.0, "True": 50.0},
    },
    "rte": {
        "train": {"entailment": 51.2, "not_entailment": 49.8},
        "validation": {"entailment": 52.7, "not_entailment": 47.3},
    },
    "mrpc": {
        "train": {"equivalent": 67.4, "not_equivalent": 32.6},
        "validation": {"equivalent": 68.4, "not_equivalent": 31.6},
    },
    "qqp": {
        "train": {"duplicate": 36.9, "not_duplicate": 63.1},
        "validation": {"duplicate": 36.8, "not_duplicate": 63.2},
    },
}


def compute_label_distribution(dataset_config: DatasetConfig, split: str = "train") -> LabelDistribution:
    """
    Compute label distribution for a dataset.

    Args:
        dataset_config: Dataset configuration
        split: Data split ('train' or 'validation')

    Returns:
        LabelDistribution object with counts and percentages
    """
    spec = get_task_spec(dataset_config.task_name)
    hf_dataset = load_dataset(
        dataset_config.dataset_name,
        spec.dataset_config_name,
        split=split,
    )

    # Extract labels from processed examples
    labels = []
    for example in hf_dataset:
        processed = spec.process(example)
        # For classification tasks, target_text is the label
        labels.append(processed.target_text.strip())

    label_counts = Counter(labels)
    total = len(labels)
    label_percentages = {label: (count / total * 100) for label, count in label_counts.items()}

    return LabelDistribution(
        task_name=dataset_config.task_name,
        split=split,
        total_examples=total,
        label_counts=dict(label_counts),
        label_percentages=label_percentages,
    )


def validate_label_distribution(
    dataset_config: DatasetConfig,
    split: str = "train",
    tolerance: float = 2.0,
) -> dict[str, bool]:
    """
    Validate computed label distribution against paper's reported values.

    Args:
        dataset_config: Dataset configuration
        split: Data split to validate
        tolerance: Tolerance in percentage points (default 2%)

    Returns:
        Dictionary with validation results for each label
    """
    computed = compute_label_distribution(dataset_config, split)

    if dataset_config.task_name not in PAPER_LABEL_DISTRIBUTIONS:
        print(f"Warning: No paper reference for {dataset_config.task_name}")
        return {}

    paper_dist = PAPER_LABEL_DISTRIBUTIONS[dataset_config.task_name]
    if split not in paper_dist:
        print(f"Warning: No paper reference for {dataset_config.task_name} {split} split")
        return {}

    results = {}
    for label, paper_pct in paper_dist[split].items():
        computed_pct = computed.label_percentages.get(label, 0.0)
        diff = abs(computed_pct - paper_pct)
        is_valid = diff <= tolerance

        results[label] = is_valid

        status = "✓" if is_valid else "✗"
        print(
            f"{status} {label:20s}: paper={paper_pct:6.1f}% "
            f"computed={computed_pct:6.1f}% (diff={diff:5.1f}%)"
        )

    return results


def print_label_distribution(dist: LabelDistribution) -> None:
    """Pretty-print label distribution."""
    print(f"\n{dist.task_name.upper()} - {dist.split.upper()}")
    print(f"Total examples: {dist.total_examples}")
    print("\nLabel Distribution:")
    for label, count in sorted(dist.label_counts.items()):
        pct = dist.label_percentages[label]
        print(f"  {label:20s}: {count:6d} ({pct:6.1f}%)")


def validate_all_superglue() -> None:
    """Validate all SuperGLUE datasets against paper's reported distributions."""
    superglue_tasks = ["boolq", "cb", "copa", "multirc", "wic", "rte", "mrpc", "qqp"]

    print("=" * 80)
    print("SUPERGLUE LABEL DISTRIBUTION VALIDATION")
    print("=" * 80)

    all_valid = True
    for task in superglue_tasks:
        print(f"\n{'=' * 80}")
        print(f"Task: {task}")
        print(f"{'=' * 80}")

        config = DatasetConfig(task_name=task, dataset_name="glue" if task in ["mrpc", "qqp"] else "super_glue")

        # Validate train split
        print("\nTrain Split:")
        train_results = validate_label_distribution(config, "train")
        if not all(train_results.values()):
            all_valid = False

        # Validate validation split
        print("\nValidation Split:")
        valid_results = validate_label_distribution(config, "validation")
        if not all(valid_results.values()):
            all_valid = False

    print(f"\n{'=' * 80}")
    if all_valid:
        print("✓ All validations passed!")
    else:
        print("✗ Some validations failed. Check differences above.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    validate_all_superglue()
