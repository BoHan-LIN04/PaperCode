from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .config import DatasetConfig
from .tasks import ProcessedExample, get_task_spec


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    processed: ProcessedExample


class Text2TextDataset(Dataset[TokenizedExample]):
    def __init__(self, examples: list[TokenizedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.examples[index]


class Seq2SeqCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[TokenizedExample]) -> dict[str, torch.Tensor | list[ProcessedExample]]:
        max_input_length = max(len(item.input_ids) for item in batch)
        max_label_length = max(len(item.labels) for item in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []
        processed = []
        for item in batch:
            input_padding = [pad_id] * (max_input_length - len(item.input_ids))
            label_padding = [-100] * (max_label_length - len(item.labels))
            input_ids.append(item.input_ids + input_padding)
            attention_mask.append(item.attention_mask + [0] * len(input_padding))
            labels.append(item.labels + label_padding)
            processed.append(item.processed)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "examples": processed,
        }


def load_task_dataset(dataset_config: DatasetConfig, tokenizer, split: str) -> tuple[Text2TextDataset, list[ProcessedExample]]:
    spec = get_task_spec(dataset_config.task_name)
    hf_dataset = load_dataset(dataset_config.dataset_name, spec.dataset_config_name, split=split)
    limit = dataset_config.max_train_examples if split == dataset_config.train_split else dataset_config.max_eval_examples
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    processed_examples: list[ProcessedExample] = [spec.process(example) for example in hf_dataset]
    tokenized_examples: list[TokenizedExample] = []
    for processed in processed_examples:
        source = tokenizer(
            processed.source_text,
            truncation=True,
            max_length=dataset_config.max_source_length,
            add_special_tokens=True,
        )
        target = tokenizer(
            processed.target_text,
            truncation=True,
            max_length=dataset_config.max_target_length,
            add_special_tokens=True,
        )
        tokenized_examples.append(
            TokenizedExample(
                input_ids=source["input_ids"],
                attention_mask=source["attention_mask"],
                labels=target["input_ids"],
                processed=processed,
            )
        )
    return Text2TextDataset(tokenized_examples), processed_examples