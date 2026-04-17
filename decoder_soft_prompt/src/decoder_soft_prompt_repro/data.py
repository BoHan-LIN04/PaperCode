from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import DatasetConfig


@dataclass
class TokenizedCausalExample:
    source_ids: list[int]
    source_attention_mask: list[int]
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    source_text: str
    target_text: str


class CausalTextDataset(Dataset[TokenizedCausalExample]):
    def __init__(self, examples: list[TokenizedCausalExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedCausalExample:
        return self.examples[index]


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[TokenizedCausalExample]) -> dict[str, torch.Tensor | list[str]]:
        pad_id = self.tokenizer.pad_token_id
        max_source_length = max(len(item.source_ids) for item in batch)
        max_full_length = max(len(item.input_ids) for item in batch)

        source_ids = []
        source_attention_mask = []
        input_ids = []
        attention_mask = []
        labels = []
        source_texts = []
        target_texts = []

        for item in batch:
            source_padding = [pad_id] * (max_source_length - len(item.source_ids))
            full_padding = [pad_id] * (max_full_length - len(item.input_ids))
            label_padding = [-100] * (max_full_length - len(item.labels))

            source_ids.append(item.source_ids + source_padding)
            source_attention_mask.append(item.source_attention_mask + [0] * len(source_padding))
            input_ids.append(item.input_ids + full_padding)
            attention_mask.append(item.attention_mask + [0] * len(full_padding))
            labels.append(item.labels + label_padding)
            source_texts.append(item.source_text)
            target_texts.append(item.target_text)

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_attention_mask": torch.tensor(source_attention_mask, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "source_texts": source_texts,
            "target_texts": target_texts,
        }


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_jsonl_dataset(dataset_config: DatasetConfig, tokenizer, split: str) -> CausalTextDataset:
    path = dataset_config.train_file if split == "train" else dataset_config.eval_file
    rows = _read_jsonl(path)
    limit = dataset_config.max_train_examples if split == "train" else dataset_config.max_eval_examples
    if limit is not None:
        rows = rows[:limit]

    examples: list[TokenizedCausalExample] = []
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for decoder-only training")

    for row in rows:
        if dataset_config.input_field not in row:
            raise ValueError(f"Missing input field '{dataset_config.input_field}' in row: {row}")
        if dataset_config.target_field not in row:
            raise ValueError(f"Missing target field '{dataset_config.target_field}' in row: {row}")

        source_text = str(row[dataset_config.input_field])
        target_text = str(row[dataset_config.target_field])
        source = tokenizer(
            source_text,
            truncation=True,
            max_length=dataset_config.max_source_length,
            add_special_tokens=True,
        )
        target = tokenizer(
            target_text,
            truncation=True,
            max_length=dataset_config.max_target_length,
            add_special_tokens=False,
        )

        source_ids = list(source["input_ids"])
        target_ids = list(target["input_ids"])
        full_ids = source_ids + target_ids + [eos_token_id]
        labels = ([-100] * len(source_ids)) + target_ids + [eos_token_id]

        examples.append(
            TokenizedCausalExample(
                source_ids=source_ids,
                source_attention_mask=[1] * len(source_ids),
                input_ids=full_ids,
                attention_mask=[1] * len(full_ids),
                labels=labels,
                source_text=source_text,
                target_text=target_text,
            )
        )

    return CausalTextDataset(examples)