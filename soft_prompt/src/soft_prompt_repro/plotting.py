from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


_METHOD_DISPLAY = {
    "model_tuning": "Model Tuning",
    "model_tuning_multitask": "Model Tuning (Multi-task)",
    "prompt_tuning": "Prompt Tuning",
    "prompt_design": "Prompt Design",
}

_METHOD_STYLE = {
    "model_tuning": {"color": "#c44e52", "marker": "o"},
    "model_tuning_multitask": {"color": "#dd8452", "marker": "o"},
    "prompt_tuning": {"color": "#55a868", "marker": "x"},
    "prompt_design": {"color": "#4c72b0", "marker": "s"},
}


def plot_figure1_from_csv(csv_path: str | Path, output_path: str | Path, title: str | None = None) -> str:
    source = Path(csv_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["model_params"] = float(row["model_params"])
            row["mean_score"] = float(row["mean_score"])
            row["std_score"] = float(row["std_score"])
            grouped[row["method"]].append(row)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for method, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["model_params"])
        xs = [item["model_params"] for item in rows]
        ys = [item["mean_score"] for item in rows]
        errs = [item["std_score"] for item in rows]
        style = _METHOD_STYLE.get(method, {"color": "#333333", "marker": "o"})
        ax.plot(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            label=_METHOD_DISPLAY.get(method, method),
        )
        if any(value > 0 for value in errs):
            lower = [y - e for y, e in zip(ys, errs)]
            upper = [y + e for y, e in zip(ys, errs)]
            ax.fill_between(xs, lower, upper, color=style["color"], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("SuperGLUE Score")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.set_title(title or "Prompt Tuning vs Model Tuning")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return str(destination)
