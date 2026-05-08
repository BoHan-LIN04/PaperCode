from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib import error, request

from sklearn.metrics import accuracy_score, classification_report, f1_score


LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenRouter base model on emotion28 single-label dataset")
    parser.add_argument("--input", default="data/emotion28_test.jsonl", help="Path to JSONL file with input/target fields")
    parser.add_argument("--output", default="artifacts/baselines/openrouter_base_metrics.json", help="Path to metrics JSON")
    parser.add_argument(
        "--predictions-output",
        default="artifacts/baselines/openrouter_base_predictions.jsonl",
        help="Path to per-example predictions JSONL",
    )
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct", help="OpenRouter model name")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of samples (0 means full file)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per request on transient failure")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key; prefer OPENROUTER_API_KEY env var")
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "input" not in row or "target" not in row:
                continue
            rows.append(row)
    return rows


def build_prompt(text: str) -> str:
    labels_str = ", ".join(LABELS)
    return (
        "Classify the emotion of the text into exactly one label from this set:\n"
        f"{labels_str}\n"
        "Return only one label word from the set and nothing else.\n\n"
        f"Text: {text}"
    )


def call_openrouter(api_key: str, model: str, prompt: str, timeout: float, max_retries: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            req = request.Request(
                url="https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"]).strip()
        except error.HTTPError as exc:
            last_error = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(1.5)
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
            continue

    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def normalize_prediction(text: str) -> str:
    lowered = text.strip().lower()
    if lowered in LABELS:
        return lowered

    first = lowered.split()[0] if lowered else ""
    if first in LABELS:
        return first

    for token in lowered.replace("/", " ").replace(",", " ").split():
        if token in LABELS:
            return token

    return "neutral"


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

    rows = load_rows(args.input)
    if not rows:
        raise ValueError(f"No valid rows found in {args.input}")

    random.seed(args.seed)
    if args.sample_size > 0 and len(rows) > args.sample_size:
        rows = random.sample(rows, args.sample_size)

    y_true: list[str] = []
    y_pred: list[str] = []
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(str(row["input"]))
        try:
            raw = call_openrouter(
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            pred = normalize_prediction(raw)
            err = None
        except Exception as exc:  # noqa: BLE001
            raw = ""
            pred = "neutral"
            err = str(exc)

        target = str(row["target"]).strip().lower()
        y_true.append(target)
        y_pred.append(pred)

        records.append(
            {
                "id": row.get("id"),
                "input": row["input"],
                "target": target,
                "prediction": pred,
                "raw_response": raw,
                "error": err,
            }
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(rows)}")

    metrics = {
        "model": args.model,
        "dataset": args.input,
        "num_samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", labels=LABELS, zero_division=0)),
        "pred_label_count": len(set(y_pred)),
        "pred_distribution": dict(Counter(y_pred).most_common()),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            digits=4,
            zero_division=0,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[RESULT] accuracy:", metrics["accuracy"])
    print("[RESULT] macro_f1:", metrics["macro_f1"])
    print("[RESULT] pred_label_count:", metrics["pred_label_count"])
    print("[RESULT] metrics_file:", str(output_path))
    print("[RESULT] predictions_file:", str(pred_path))


if __name__ == "__main__":
    main()
