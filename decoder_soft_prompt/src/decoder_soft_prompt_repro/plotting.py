import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

METHOD_DISPLAY = {
    "model_tuning": "Model Tuning",
    "model_tuning_multitask": "Model Tuning (Multi-task)",
    "prompt_tuning": "Prompt Tuning",
    "prompt_design": "Prompt Design",
    "emotion_vectors": "Emotion Vectors",
}

METHOD_STYLE = {
    "model_tuning": {"color": "#c44e52", "marker": "o"},
    "model_tuning_multitask": {"color": "#dd8452", "marker": "o"},
    "prompt_tuning": {"color": "#55a868", "marker": "x"},
    "prompt_design": {"color": "#4c72b0", "marker": "s"},
    "emotion_vectors": {"color": "#9370DB", "marker": "^"},
}

def plot_figure_from_csv(csv_path: str | Path, output_path: str | Path, title: str | None = None) -> str:
    source = Path(csv_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["x"] = float(row.get("model_params", row.get("step", 0)))
            row["mean_score"] = float(row.get("mean_score", row.get("eval_loss", 0)))
            row["std_score"] = float(row.get("std_score", 0))
            grouped[row.get("method", "prompt_tuning")].append(row)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for method, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["x"])
        xs = [item["x"] for item in rows]
        ys = [item["mean_score"] for item in rows]
        errs = [item["std_score"] for item in rows]
        style = METHOD_STYLE.get(method, {"color": "#333333", "marker": "o"})
        ax.plot(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            label=METHOD_DISPLAY.get(method, method),
        )
        if any(value > 0 for value in errs):
            lower = [y - e for y, e in zip(ys, errs)]
            upper = [y + e for y, e in zip(ys, errs)]
            ax.fill_between(xs, lower, upper, color=style["color"], alpha=0.15)

    ax.set_xlabel("Model Params / Step")
    ax.set_ylabel("Score / Loss")
    ax.legend(loc="best")
    ax.set_title(title or "Prompt Tuning Results")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return str(destination)
