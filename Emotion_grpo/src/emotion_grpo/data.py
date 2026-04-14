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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _build_logical_messages(question: str, prompt_style: str = "answer_only") -> list[dict[str, str]]:
    if prompt_style == "reasoning":
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful logical reasoning assistant. Think through the problem step by step and then "
                    "end with a final answer on the last line in the form `Final answer: ...`."
                ),
            },
            {
                "role": "user",
                "content": question.strip() + "\n\nReason carefully before giving the final answer.",
            },
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are a careful logical reasoning assistant. Solve the task and return only the final short answer. "
                "Do not include explanations, derivations, or extra commentary."
            ),
        },
        {
            "role": "user",
            "content": question.strip() + "\n\nReturn only the final answer.",
        },
    ]


def _string_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def prepare_logical_qa_jsonl(
    source_files: list[str] | list[Path],
    output_path: str | Path,
    split_name: str,
    prompt_style: str = "answer_only",
    overwrite: bool = False,
) -> Path:
    target = Path(output_path)
    if target.exists() and not overwrite:
        return target

    rows: list[dict[str, Any]] = []
    running_index = 0
    for source_file in source_files:
        source_path = Path(source_file)
        dataset_name = source_path.parent.name
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                question = str(raw.get("question", "")).strip()
                answer = raw.get("answer")
                if not question or answer in (None, ""):
                    continue
                running_index += 1
                metadata = {
                    "id": _string_or_none(f"{dataset_name}-{raw.get('id', running_index)}"),
                    "ground_truth": _string_or_none(str(answer).strip()),
                    "label": _string_or_none(str(answer).strip()),
                    "data_source": "logical_qa",
                    "ability": "logical_reasoning",
                    "logical_dataset": dataset_name,
                    "domain": _string_or_none(raw.get("domain")),
                    "year": _string_or_none(raw.get("year")),
                    "level": _string_or_none(raw.get("level")),
                    "task_id": _string_or_none(raw.get("task_id")),
                    "entry_point": _string_or_none(raw.get("entry_point")),
                    "is_good": _string_or_none(raw.get("is_good")),
                }
                rows.append(
                    {
                        "messages": _build_logical_messages(question, prompt_style=prompt_style),
                        "metadata": metadata,
                    }
                )

    if not rows:
        raise ValueError(f"No usable logical QA rows were found for split {split_name}")
    return _write_jsonl(target, rows)


def ensure_config_jsonl(config: dict[str, Any], overwrite: bool = False) -> tuple[Path, Path]:
    datasets = config["datasets"]
    logical_cfg = datasets.get("logical_qa_sources") or {}
    train_jsonl = Path(datasets["train_jsonl"])
    val_jsonl = Path(datasets["val_jsonl"])
    if logical_cfg:
        prompt_style = str(logical_cfg.get("prompt_style", "answer_only"))
        prepare_logical_qa_jsonl(
            source_files=logical_cfg.get("train_files", []),
            output_path=train_jsonl,
            split_name="train",
            prompt_style=prompt_style,
            overwrite=overwrite,
        )
        prepare_logical_qa_jsonl(
            source_files=logical_cfg.get("val_files", []),
            output_path=val_jsonl,
            split_name="val",
            prompt_style=prompt_style,
            overwrite=overwrite,
        )
    return train_jsonl, val_jsonl


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
    train_jsonl, val_jsonl = ensure_config_jsonl(config=config, overwrite=overwrite)
    train_parquet = convert_jsonl_to_parquet(
        jsonl_path=train_jsonl,
        parquet_path=datasets["processed_train"],
        split_name="train",
        overwrite=overwrite,
    )
    val_parquet = convert_jsonl_to_parquet(
        jsonl_path=val_jsonl,
        parquet_path=datasets["processed_val"],
        split_name="val",
        overwrite=overwrite,
    )
    return train_parquet, val_parquet
