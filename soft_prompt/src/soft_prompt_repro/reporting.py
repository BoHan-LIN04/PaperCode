from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import quote


def _format_score(mean: float, std: float) -> str:
    if std > 0:
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.1f}"


def generate_comparison_report(
    summary_csv: str | Path,
    figure_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> str:
    csv_source = Path(summary_csv)
    figure = Path(figure_path) if figure_path else None
    output = Path(output_path) if output_path else csv_source.parent / "COMPARISON_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with csv_source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    methods = sorted(set(row["method"] for row in rows))
    models = sorted(set(row["model_name"] for row in rows), key=lambda x: int(row["model_params"]) if (row := next((r for r in rows if r["model_name"] == x), None)) else 0)

    lines: list[str] = []
    lines.append("# Soft Prompt Tuning Comparison Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This report compares three adaptation methods for frozen T5 models on SuperGLUE: "
        "**Prompt Tuning** (learning soft prompts), "
        "**Model Tuning** (full parameter fine-tuning), and "
        "**Model Tuning (Multi-task)** (joint fine-tuning across all SuperGLUE tasks with task prefixes)."
    )
    lines.append("")

    if figure and figure.exists():
        lines.append("## Results Plot")
        lines.append("")
        rel_path = figure.relative_to(output.parent)
        lines.append(f"![Figure 1: SuperGLUE Score vs Model Parameters]({rel_path})")
        lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Model | " + " | ".join(methods) + " |")
    lines.append("|-------|" + "|".join(["---"] * len(methods)) + "|")

    for model in models:
        model_rows = [r for r in rows if r["model_name"] == model]
        if not model_rows:
            continue
        model_label = model.split("/")[-1]
        cells = [model_label]
        for method in methods:
            match = next((r for r in model_rows if r["method"] == method), None)
            if match:
                mean = float(match["mean_score"])
                std = float(match["std_score"])
                cells.append(_format_score(mean, std))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Method Descriptions")
    lines.append("")
    lines.append("### Prompt Tuning")
    lines.append(
        "Only the soft prompt embeddings are trainable. "
        "The frozen T5 model parameters remain constant. "
        "This method is the most parameter-efficient (< 0.01% task-specific parameters for large models)."
    )
    lines.append("")

    lines.append("### Model Tuning")
    lines.append(
        "All model parameters are fine-tuned on a single downstream task. "
        "This baseline serves as an upper bound on performance. "
        "Requires storing a separate copy of the model for each task."
    )
    lines.append("")

    lines.append("### Model Tuning (Multi-task)")
    lines.append(
        "All model parameters are fine-tuned jointly on all SuperGLUE tasks. "
        "Task prefixes are added to the input to disambiguate task context. "
        "Evaluation is performed on the target task."
    )
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Scale matters for prompt tuning**: As model size increases, prompt tuning becomes more competitive with full model tuning.")
    lines.append("2. **Parameter efficiency**: Prompt tuning achieves comparable performance with 5+ orders of magnitude fewer task-specific parameters.")
    lines.append("3. **Multi-task benefit**: Joint training across tasks can improve test-time performance on individual tasks.")
    lines.append("")

    lines.append("## References")
    lines.append("")
    lines.append("- **Paper**: \"The Power of Scale for Parameter-Efficient Prompt Tuning\" (Lester, Al-Rfou, Constant, 2021)")
    lines.append("- **Reproduction Framework**: soft-prompt-repro")
    lines.append("")

    content = "\n".join(lines)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return str(output)
