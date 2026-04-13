from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _validate_message(message: dict[str, Any], record_index: int) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"Record {record_index}: each message must be an object")
    if "role" not in message or "content" not in message:
        raise ValueError(f"Record {record_index}: each message must contain role and content")


def normalize_jsonl_record(raw_record: dict[str, Any], record_index: int) -> dict[str, Any]:
    messages = raw_record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {record_index}: messages must be a non-empty list")
    for message in messages:
        _validate_message(message, record_index)

    metadata = raw_record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Record {record_index}: metadata must be an object when provided")

    return {"messages": messages, "metadata": metadata}


def _to_verl_row(record: dict[str, Any], split_name: str, record_index: int) -> dict[str, Any]:
    metadata = dict(record["metadata"])
    record_id = metadata.get("id", f"{split_name}-{record_index}")
    data_source = metadata.get("data_source", "emotion_intrinsic")
    ability = metadata.get("ability", "emotion_rl")
    # Support both the newer `ground_truth` field and the older `label` field.
    ground_truth = metadata.get("ground_truth", metadata.get("label", ""))

    extra_info = {
        "messages": record["messages"],
        "metadata": metadata,
        "split": split_name,
        "record_id": record_id,
    }

    return {
        "data_source": data_source,
        "prompt": record["messages"],
        "ability": ability,
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def convert_jsonl_to_parquet(
    jsonl_path: str | Path,
    parquet_path: str | Path,
    split_name: str,
    overwrite: bool = False,
) -> Path:
    source = Path(jsonl_path)
    target = Path(parquet_path)
    if target.exists() and not overwrite:
        return target

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw_record = json.loads(stripped)
            normalized = normalize_jsonl_record(raw_record, record_index=index)
            rows.append(_to_verl_row(normalized, split_name=split_name, record_index=index))

    if not rows:
        raise ValueError(f"No usable rows were found in {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def ensure_demo_parquet(config: dict[str, Any], overwrite: bool = False) -> tuple[Path, Path]:
    datasets = config["datasets"]
    train_parquet = convert_jsonl_to_parquet(
        jsonl_path=datasets["train_jsonl"],
        parquet_path=datasets["processed_train"],
        split_name="train",
        overwrite=overwrite,
    )
    val_parquet = convert_jsonl_to_parquet(
        jsonl_path=datasets["val_jsonl"],
        parquet_path=datasets["processed_val"],
        split_name="val",
        overwrite=overwrite,
    )
    return train_parquet, val_parquet
