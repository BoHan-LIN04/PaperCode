from __future__ import annotations

import json
import random
import csv
import statistics
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

from .config import ExperimentConfig, config_to_dict
from .data import Seq2SeqCollator, Text2TextDataset, TokenizedExample, load_task_dataset
from .metrics import compute_metrics
from .plotting import plot_figure1_from_csv
from .prompt_tuning import SoftPromptT5
from .reporting import generate_comparison_report
from .tasks import canonicalize_class_prediction, get_task_spec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_adafactor_optimizer(parameters, learning_rate: float, weight_decay: float, parameter_scaling: bool) -> Adafactor:
    return Adafactor(
        parameters,
        lr=learning_rate,
        relative_step=False,
        scale_parameter=parameter_scaling,
        warmup_init=False,
        weight_decay=weight_decay,
    )


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    current_step: int,
    warmup_steps: int,
    decay_factor: float | None,
    steps_per_decay: int | None,
) -> float:
    if current_step <= 0:
        return base_lr

    lr = base_lr
    if warmup_steps > 0 and current_step <= warmup_steps:
        lr = base_lr * (current_step / warmup_steps)

    if decay_factor is not None and steps_per_decay is not None and steps_per_decay > 0:
        decay_start = max(current_step - max(warmup_steps, 0), 0)
        decay_count = decay_start // steps_per_decay
        lr = lr * (decay_factor ** decay_count)

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return model, tokenizer


def _decode_predictions(task_name: str, task_labels: list[str] | None, tokenizer, generated: torch.Tensor) -> list[str]:
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    if task_labels:
        return [canonicalize_class_prediction(item, task_labels) for item in decoded]
    return [" ".join(item.strip().split()) for item in decoded]


def _evaluate_with_generate_fn(
    task_name: str,
    task_labels: list[str] | None,
    eval_loader: DataLoader,
    eval_examples: list,
    generate_fn,
    tokenizer,
    device: torch.device,
    desc: str,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[dict[str, float], list[str]]:
    predictions: list[str] = []
    for batch in tqdm(eval_loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = generate_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        predictions.extend(_decode_predictions(task_name, task_labels, tokenizer, generated))
    metrics = compute_metrics(task_name, predictions, eval_examples)
    return metrics, predictions


def evaluate_prompt_model(config: ExperimentConfig, prompt_path: str | Path | None = None) -> dict:
    set_seed(config.training.seed)
    device = resolve_device(config.model.device)
    task_spec = get_task_spec(config.dataset.task_name)
    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    prompt_model = SoftPromptT5(
        model=model,
        num_virtual_tokens=config.prompt.num_virtual_tokens,
        init_strategy=config.prompt.init_strategy,
        random_range=config.prompt.random_range,
        sampled_vocab_size=config.prompt.sampled_vocab_size,
        tokenizer=tokenizer,
        label_texts=task_spec.label_texts,
    ).to(device)
    if prompt_path is not None:
        prompt_model.load_prompt(prompt_path)

    eval_dataset, examples = load_task_dataset(config.dataset, tokenizer, config.dataset.eval_split)
    collator = Seq2SeqCollator(tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    prompt_model.eval()
    metrics, predictions = _evaluate_with_generate_fn(
        task_name=config.dataset.task_name,
        task_labels=task_spec.label_texts if task_spec.is_classification else None,
        eval_loader=eval_loader,
        eval_examples=examples,
        generate_fn=prompt_model.generate,
        tokenizer=tokenizer,
        device=device,
        desc="eval",
        max_new_tokens=config.training.generation_max_new_tokens,
        num_beams=config.training.num_beams,
    )
    return {"metrics": metrics, "predictions": predictions}


def train_prompt_model(config: ExperimentConfig) -> dict:
    set_seed(config.training.seed)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config_to_dict(config))

    device = resolve_device(config.model.device)
    task_spec = get_task_spec(config.dataset.task_name)
    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    prompt_model = SoftPromptT5(
        model=model,
        num_virtual_tokens=config.prompt.num_virtual_tokens,
        init_strategy=config.prompt.init_strategy,
        random_range=config.prompt.random_range,
        sampled_vocab_size=config.prompt.sampled_vocab_size,
        tokenizer=tokenizer,
        label_texts=task_spec.label_texts,
    ).to(device)

    train_dataset, _ = load_task_dataset(config.dataset, tokenizer, config.dataset.train_split)
    eval_dataset, eval_examples = load_task_dataset(config.dataset, tokenizer, config.dataset.eval_split)
    collator = Seq2SeqCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = _build_adafactor_optimizer(
        prompt_model.trainable_parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        parameter_scaling=config.training.parameter_scaling,
    )

    best_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    best_prompt_path = output_dir / "best_prompt.pt"
    patience = 0
    step = 0
    train_iterator = iter(train_loader)
    history: list[dict[str, float]] = []

    while step < config.training.max_steps:
        prompt_model.train()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        current_lr = _set_optimizer_lr(
            optimizer,
            base_lr=config.training.learning_rate,
            current_step=step + 1,
            warmup_steps=config.training.warmup_steps,
            decay_factor=config.training.decay_factor,
            steps_per_decay=config.training.steps_per_decay,
        )

        outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.trainable_parameters(), config.training.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % config.training.logging_steps == 0:
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), "learning_rate": current_lr})

        if step % config.training.eval_steps != 0 and step < config.training.max_steps:
            continue

        prompt_model.eval()
        metrics, predictions = _evaluate_with_generate_fn(
            task_name=config.dataset.task_name,
            task_labels=task_spec.label_texts if task_spec.is_classification else None,
            eval_loader=eval_loader,
            eval_examples=eval_examples,
            generate_fn=prompt_model.generate,
            tokenizer=tokenizer,
            device=device,
            desc=f"eval@{step}",
            max_new_tokens=config.training.generation_max_new_tokens,
            num_beams=config.training.num_beams,
        )
        current_score = metrics["score"]
        history.append({"step": step, **metrics})
        if current_score > best_metric:
            best_metric = current_score
            best_metrics = metrics
            patience = 0
            prompt_model.save_prompt(best_prompt_path, metadata={"step": step, "metrics": metrics})
            if config.output.save_predictions:
                _write_json(output_dir / "best_predictions.json", {"predictions": predictions})
        else:
            patience += 1
        if patience >= config.training.early_stopping_patience:
            break

    _write_json(output_dir / "history.json", {"history": history})
    if config.output.save_metrics:
        _write_json(output_dir / "metrics.json", best_metrics)
    return {
        "output_dir": str(output_dir),
        "best_prompt_path": str(best_prompt_path),
        "metrics": best_metrics,
        "best_score": best_metric,
    }


def train_model_tuning(config: ExperimentConfig) -> dict:
    set_seed(config.training.seed)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config_to_dict(config))

    device = resolve_device(config.model.device)
    task_spec = get_task_spec(config.dataset.task_name)
    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    model.to(device)
    model.train()

    train_dataset, _ = load_task_dataset(config.dataset, tokenizer, config.dataset.train_split)
    eval_dataset, eval_examples = load_task_dataset(config.dataset, tokenizer, config.dataset.eval_split)
    collator = Seq2SeqCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = _build_adafactor_optimizer(
        model.parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        parameter_scaling=config.training.parameter_scaling,
    )

    best_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    best_model_dir = output_dir / "best_model"
    patience = 0
    step = 0
    train_iterator = iter(train_loader)
    history: list[dict[str, float]] = []

    while step < config.training.max_steps:
        model.train()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        current_lr = _set_optimizer_lr(
            optimizer,
            base_lr=config.training.learning_rate,
            current_step=step + 1,
            warmup_steps=config.training.warmup_steps,
            decay_factor=config.training.decay_factor,
            steps_per_decay=config.training.steps_per_decay,
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % config.training.logging_steps == 0:
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), "learning_rate": current_lr})

        if step % config.training.eval_steps != 0 and step < config.training.max_steps:
            continue

        model.eval()
        metrics, predictions = _evaluate_with_generate_fn(
            task_name=config.dataset.task_name,
            task_labels=task_spec.label_texts if task_spec.is_classification else None,
            eval_loader=eval_loader,
            eval_examples=eval_examples,
            generate_fn=model.generate,
            tokenizer=tokenizer,
            device=device,
            desc=f"eval@{step}",
            max_new_tokens=config.training.generation_max_new_tokens,
            num_beams=config.training.num_beams,
        )
        current_score = metrics["score"]
        history.append({"step": step, **metrics})
        if current_score > best_metric:
            best_metric = current_score
            best_metrics = metrics
            patience = 0
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            if config.output.save_predictions:
                _write_json(output_dir / "best_predictions.json", {"predictions": predictions})
        else:
            patience += 1
        if patience >= config.training.early_stopping_patience:
            break

    _write_json(output_dir / "history.json", {"history": history})
    if config.output.save_metrics:
        _write_json(output_dir / "metrics.json", best_metrics)
    return {
        "output_dir": str(output_dir),
        "best_model_path": str(best_model_dir),
        "metrics": best_metrics,
        "best_score": best_metric,
    }


def _with_task_prefix(dataset: Text2TextDataset, task_name: str, tokenizer, max_source_length: int) -> Text2TextDataset:
    prefixed: list[TokenizedExample] = []
    prefix = f"task: {task_name} "
    for item in dataset.examples:
        source = tokenizer(
            prefix + item.processed.source_text,
            truncation=True,
            max_length=max_source_length,
            add_special_tokens=True,
        )
        prefixed.append(
            TokenizedExample(
                input_ids=source["input_ids"],
                attention_mask=source["attention_mask"],
                labels=item.labels,
                processed=item.processed,
            )
        )
    return Text2TextDataset(prefixed)


def train_model_tuning_multitask(config: ExperimentConfig) -> dict:
    set_seed(config.training.seed)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config_to_dict(config))

    device = resolve_device(config.model.device)
    target_task_spec = get_task_spec(config.dataset.task_name)
    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    model.to(device)
    model.train()

    all_train_examples: list[TokenizedExample] = []
    tasks = list(config.baseline.multitask_tasks)
    if not tasks:
        raise ValueError("baseline.multitask_tasks must not be empty")
    for task_name in tasks:
        task_dataset_config = deepcopy(config.dataset)
        task_dataset_config.task_name = task_name
        task_train_dataset, _ = load_task_dataset(task_dataset_config, tokenizer, task_dataset_config.train_split)
        if config.baseline.add_task_prefix:
            task_train_dataset = _with_task_prefix(
                task_train_dataset,
                task_name=task_name,
                tokenizer=tokenizer,
                max_source_length=config.dataset.max_source_length,
            )
        all_train_examples.extend(task_train_dataset.examples)

    train_dataset = Text2TextDataset(all_train_examples)
    eval_dataset, eval_examples = load_task_dataset(config.dataset, tokenizer, config.dataset.eval_split)
    if config.baseline.add_task_prefix:
        eval_dataset = _with_task_prefix(
            eval_dataset,
            task_name=config.dataset.task_name,
            tokenizer=tokenizer,
            max_source_length=config.dataset.max_source_length,
        )
    collator = Seq2SeqCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = _build_adafactor_optimizer(
        model.parameters(),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        parameter_scaling=config.training.parameter_scaling,
    )

    best_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    best_model_dir = output_dir / "best_model"
    patience = 0
    step = 0
    train_iterator = iter(train_loader)
    history: list[dict[str, float]] = []

    while step < config.training.max_steps:
        model.train()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        current_lr = _set_optimizer_lr(
            optimizer,
            base_lr=config.training.learning_rate,
            current_step=step + 1,
            warmup_steps=config.training.warmup_steps,
            decay_factor=config.training.decay_factor,
            steps_per_decay=config.training.steps_per_decay,
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % config.training.logging_steps == 0:
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), "learning_rate": current_lr})

        if step % config.training.eval_steps != 0 and step < config.training.max_steps:
            continue

        model.eval()
        metrics, predictions = _evaluate_with_generate_fn(
            task_name=config.dataset.task_name,
            task_labels=target_task_spec.label_texts if target_task_spec.is_classification else None,
            eval_loader=eval_loader,
            eval_examples=eval_examples,
            generate_fn=model.generate,
            tokenizer=tokenizer,
            device=device,
            desc=f"eval@{step}",
            max_new_tokens=config.training.generation_max_new_tokens,
            num_beams=config.training.num_beams,
        )
        current_score = metrics["score"]
        history.append({"step": step, **metrics})
        if current_score > best_metric:
            best_metric = current_score
            best_metrics = metrics
            patience = 0
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            if config.output.save_predictions:
                _write_json(output_dir / "best_predictions.json", {"predictions": predictions})
        else:
            patience += 1
        if patience >= config.training.early_stopping_patience:
            break

    _write_json(output_dir / "history.json", {"history": history})
    if config.output.save_metrics:
        _write_json(output_dir / "metrics.json", best_metrics)
    return {
        "output_dir": str(output_dir),
        "best_model_path": str(best_model_dir),
        "metrics": best_metrics,
        "best_score": best_metric,
    }


def run_sweep(config: ExperimentConfig) -> dict:
    if not config.sweep.enabled:
        raise ValueError("Sweep requested but config.sweep.enabled is false")

    root_output = Path(config.output.output_dir)
    results = []
    for value in config.sweep.values:
        run_config = deepcopy(config)
        target = run_config
        parts = config.sweep.parameter.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
        run_name = str(value).replace("/", "_")
        run_config.output.output_dir = str(root_output / run_name)
        results.append({"value": value, **train_prompt_model(run_config)})

    summary = {"parameter": config.sweep.parameter, "results": results}
    _write_json(root_output / "sweep_summary.json", summary)
    return summary


def ensemble_prompt_models(config: ExperimentConfig, prompt_paths: list[str]) -> dict:
    if not prompt_paths:
        raise ValueError("At least one prompt path is required for ensembling")
    task_spec = get_task_spec(config.dataset.task_name)
    device = resolve_device(config.model.device)
    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    eval_dataset, examples = load_task_dataset(config.dataset, tokenizer, config.dataset.eval_split)
    collator = Seq2SeqCollator(tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    all_predictions: list[list[str]] = []
    for prompt_path in prompt_paths:
        prompt_model = SoftPromptT5(
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            num_virtual_tokens=config.prompt.num_virtual_tokens,
            init_strategy=config.prompt.init_strategy,
            random_range=config.prompt.random_range,
            sampled_vocab_size=config.prompt.sampled_vocab_size,
            tokenizer=tokenizer,
            label_texts=task_spec.label_texts,
        ).to(device)
        prompt_model.load_prompt(prompt_path)
        prompt_model.eval()
        predictions = []
        for batch in tqdm(eval_loader, desc=f"ensemble:{Path(prompt_path).name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = prompt_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.training.generation_max_new_tokens,
                num_beams=config.training.num_beams,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if task_spec.is_classification and task_spec.label_texts:
                decoded = [canonicalize_class_prediction(item, task_spec.label_texts) for item in decoded]
            else:
                decoded = [" ".join(item.strip().split()) for item in decoded]
            predictions.extend(decoded)
        all_predictions.append(predictions)

    majority_predictions = []
    for row in zip(*all_predictions):
        counts = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        majority_predictions.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])

    metrics = compute_metrics(config.dataset.task_name, majority_predictions, examples)
    return {"metrics": metrics, "predictions": majority_predictions}


def adapt_language_model(config: ExperimentConfig) -> dict:
    if not config.adaptation.enabled:
        raise ValueError("LM adaptation requested but config.adaptation.enabled is false")
    set_seed(config.training.seed)
    device = resolve_device(config.model.device)
    output_dir = Path(config.adaptation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    model, tokenizer = _load_model_and_tokenizer(model_name)
    model.to(device)

    raw_train = load_dataset(
        config.adaptation.dataset_name,
        config.adaptation.dataset_config_name,
        split=config.adaptation.train_split,
    )
    text_column = config.adaptation.text_column
    texts = [text for text in raw_train[text_column] if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError("LM adaptation dataset produced no usable text rows")

    def split_example(text: str) -> tuple[str, str]:
        tokens = text.split()
        if len(tokens) < 4:
            midpoint = max(1, len(tokens) // 2)
        else:
            midpoint = max(1, min(len(tokens) - 1, int(len(tokens) * config.adaptation.prefix_fraction)))
        return " ".join(tokens[:midpoint]), " ".join(tokens[midpoint:])

    encoded = []
    for text in texts[: config.adaptation.max_steps * config.adaptation.batch_size * 2]:
        source_text, target_text = split_example(text)
        if not source_text or not target_text:
            continue
        source = tokenizer(source_text, truncation=True, max_length=config.adaptation.max_source_length)
        target = tokenizer(target_text, truncation=True, max_length=config.adaptation.max_target_length)
        encoded.append({
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
        })
    if not encoded:
        raise ValueError("LM adaptation preprocessing produced no valid examples")

    class AdaptationDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(items):
        max_input = max(len(item["input_ids"]) for item in items)
        max_label = max(len(item["labels"]) for item in items)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in items:
            input_padding = [tokenizer.pad_token_id] * (max_input - len(item["input_ids"]))
            label_padding = [-100] * (max_label - len(item["labels"]))
            batch["input_ids"].append(item["input_ids"] + input_padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * len(input_padding))
            batch["labels"].append(item["labels"] + label_padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    loader = DataLoader(
        AdaptationDataset(encoded),
        batch_size=config.adaptation.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = Adafactor(
        model.parameters(),
        lr=config.adaptation.learning_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        weight_decay=config.adaptation.weight_decay,
    )

    iterator = iter(loader)
    losses = []
    for step in range(1, config.adaptation.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    payload = {"output_dir": str(output_dir), "mean_loss": float(np.mean(losses)) if losses else 0.0}
    _write_json(output_dir / "adaptation_metrics.json", payload)
    return payload


T5_PARAMETER_COUNT = {
    "google/t5-v1_1-small": 77_000_000,
    "google/t5-v1_1-base": 250_000_000,
    "google/t5-v1_1-large": 800_000_000,
    "google/t5-v1_1-xl": 3_000_000_000,
    "google/t5-v1_1-xxl": 11_000_000_000,
}


def _method_runner(method: str):
    if method == "prompt_tuning":
        return train_prompt_model
    if method == "model_tuning":
        return train_model_tuning
    if method == "model_tuning_multitask":
        return train_model_tuning_multitask
    raise ValueError(f"Unsupported compare method: {method}")


def run_model_comparison(config: ExperimentConfig) -> dict:
    if not config.compare.enabled:
        raise ValueError("Compare requested but compare.enabled is false")

    root = Path(config.output.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []

    for model_name in config.compare.model_names:
        for method in config.compare.methods:
            runner = _method_runner(method)
            for seed in config.compare.seeds:
                run_config = deepcopy(config)
                run_config.model.name_or_path = model_name
                run_config.training.seed = int(seed)
                run_config.output.output_dir = str(
                    root
                    / "runs"
                    / method
                    / model_name.replace("/", "_")
                    / f"seed_{seed}"
                )
                result = runner(run_config)
                run_rows.append(
                    {
                        "model_name": model_name,
                        "method": method,
                        "seed": seed,
                        "score": float(result.get("best_score", 0.0)),
                        "metrics": result.get("metrics", {}),
                        "output_dir": result.get("output_dir", run_config.output.output_dir),
                    }
                )

    _write_json(root / config.compare.output_json, {"runs": run_rows})

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in run_rows:
        key = (row["model_name"], row["method"])
        grouped.setdefault(key, []).append(float(row["score"]))

    summary_rows: list[dict] = []
    for (model_name, method), values in sorted(grouped.items()):
        mean_score = statistics.mean(values) if values else 0.0
        std_score = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "model_name": model_name,
                "method": method,
                "model_params": T5_PARAMETER_COUNT.get(model_name, 0),
                "mean_score": mean_score,
                "std_score": std_score,
                "n_runs": len(values),
            }
        )

    csv_path = root / config.compare.output_csv
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "method", "model_params", "mean_score", "std_score", "n_runs"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    figure_path = root / "figure1.png"
    plot_figure1_from_csv(csv_path, figure_path, title="Prompt Tuning vs Model Tuning")

    report_path = generate_comparison_report(csv_path, figure_path)

    return {
        "output_dir": str(root),
        "runs_json": str(root / config.compare.output_json),
        "summary_csv": str(csv_path),
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "rows": summary_rows,
    }


def render_figure1_plot(summary_csv: str | Path, output_path: str | Path) -> dict:
    rendered = plot_figure1_from_csv(summary_csv, output_path, title="Prompt Tuning vs Model Tuning")
    return {"summary_csv": str(summary_csv), "figure_path": rendered}