from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig, PromptConfig, config_to_dict
from .data import CausalCollator, load_jsonl_dataset
from .prompt_tuning import SoftPromptCausalLM, _load_json, _load_tensor_payload, _resolve_hidden_size


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


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def validate_prompt_configuration(prompt: PromptConfig, *, model_hidden_size: int | None = None) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {
        "init_strategy": prompt.init_strategy,
        "emotion_vector_route": prompt.emotion_vector_route,
    }

    if prompt.num_virtual_tokens <= 0:
        errors.append("prompt.num_virtual_tokens must be positive")

    if prompt.init_strategy == "from_file":
        if not prompt.prompt_path:
            errors.append("prompt.prompt_path is required when prompt.init_strategy=from_file")
        elif not Path(prompt.prompt_path).exists():
            errors.append(f"prompt.prompt_path does not exist: {prompt.prompt_path}")
        elif model_hidden_size is not None:
            try:
                prompt_tensor = _load_tensor_payload(prompt.prompt_path)
                details["prompt_file_shape"] = list(prompt_tensor.shape)
                expected_shape = (prompt.num_virtual_tokens, model_hidden_size)
                if tuple(prompt_tensor.shape) != expected_shape:
                    errors.append(
                        "prompt file shape mismatch: "
                        f"expected {expected_shape}, got {tuple(prompt_tensor.shape)}"
                    )
            except Exception as exc:
                errors.append(f"failed to load prompt.prompt_path: {exc}")

    if prompt.init_strategy == "emotion_vectors":
        route = str(prompt.emotion_vector_route).strip().lower()
        if route not in {"same_model", "projected"}:
            errors.append("prompt.emotion_vector_route must be either 'same_model' or 'projected'")
        if not prompt.emotion_vectors_path:
            errors.append("prompt.emotion_vectors_path is required when prompt.init_strategy=emotion_vectors")
        elif not Path(prompt.emotion_vectors_path).exists():
            errors.append(f"prompt.emotion_vectors_path does not exist: {prompt.emotion_vectors_path}")
        if not prompt.emotion_vector_metadata_path:
            errors.append("prompt.emotion_vector_metadata_path is required when prompt.init_strategy=emotion_vectors")
        elif not Path(prompt.emotion_vector_metadata_path).exists():
            errors.append(
                f"prompt.emotion_vector_metadata_path does not exist: {prompt.emotion_vector_metadata_path}"
            )
        if not prompt.emotion_names:
            errors.append("prompt.emotion_names must contain at least one emotion when using emotion_vectors")

        vector_width: int | None = None
        if prompt.emotion_vectors_path and Path(prompt.emotion_vectors_path).exists():
            try:
                vectors = _load_tensor_payload(prompt.emotion_vectors_path)
                details["emotion_vectors_shape"] = list(vectors.shape)
                vector_width = int(vectors.shape[1])
            except Exception as exc:
                errors.append(f"failed to load prompt.emotion_vectors_path: {exc}")

        if prompt.emotion_vector_metadata_path and Path(prompt.emotion_vector_metadata_path).exists() and prompt.emotion_names:
            try:
                metadata = _load_json(prompt.emotion_vector_metadata_path)
                available_names = [str(name).strip().lower() for name in metadata.get("emotion_names", [])]
                missing = [str(name).strip().lower() for name in prompt.emotion_names if str(name).strip().lower() not in available_names]
                details["emotion_count_in_metadata"] = len(available_names)
                if missing:
                    errors.append(f"prompt.emotion_names not found in metadata: {missing}")
            except Exception as exc:
                errors.append(f"failed to read prompt.emotion_vector_metadata_path: {exc}")

        if route == "same_model":
            if prompt.emotion_vector_projection_path:
                errors.append(
                    "prompt.emotion_vector_projection_path must be empty when prompt.emotion_vector_route=same_model"
                )
            if model_hidden_size is not None and vector_width is not None and vector_width != model_hidden_size:
                errors.append(
                    "same_model route requires matching hidden sizes: "
                    f"emotion_vector_width={vector_width}, model_hidden_size={model_hidden_size}"
                )

        if route == "projected":
            if not prompt.emotion_vector_projection_path:
                errors.append(
                    "prompt.emotion_vector_projection_path is required when prompt.emotion_vector_route=projected"
                )
            elif not Path(prompt.emotion_vector_projection_path).exists():
                errors.append(
                    f"prompt.emotion_vector_projection_path does not exist: {prompt.emotion_vector_projection_path}"
                )
            elif model_hidden_size is not None and vector_width is not None:
                try:
                    projection = _load_tensor_payload(prompt.emotion_vector_projection_path)
                    details["projection_shape"] = list(projection.shape)
                    expected_shape = (vector_width, model_hidden_size)
                    if tuple(projection.shape) != expected_shape:
                        errors.append(
                            "projection shape mismatch: "
                            f"expected {expected_shape}, got {tuple(projection.shape)}"
                        )
                except Exception as exc:
                    errors.append(f"failed to load prompt.emotion_vector_projection_path: {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def validate_experiment_config(config: ExperimentConfig) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {
        "model_name_or_path": config.model.name_or_path,
        "tokenizer_name_or_path": config.model.tokenizer_name_or_path or config.model.name_or_path,
        "device": config.model.device,
    }

    for field_name in ["train_file", "eval_file"]:
        path = Path(getattr(config.dataset, field_name))
        if not path.exists():
            errors.append(f"dataset.{field_name} does not exist: {path}")
        else:
            checks[f"dataset_{field_name}"] = str(path)

    if config.training.batch_size <= 0:
        errors.append("training.batch_size must be positive")
    if config.training.eval_batch_size <= 0:
        errors.append("training.eval_batch_size must be positive")
    if config.training.max_steps <= 0:
        errors.append("training.max_steps must be positive")
    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be positive")

    model_hidden_size: int | None = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name_or_path or config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
            use_fast=False,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is None:
            errors.append("tokenizer has neither pad_token nor eos_token")
        elif tokenizer.pad_token is None:
            warnings.append("tokenizer.pad_token is missing; runtime will fall back to eos_token")

        model_kwargs = {"trust_remote_code": config.model.trust_remote_code}
        if config.model.torch_dtype:
            model_kwargs["dtype"] = getattr(torch, config.model.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, **model_kwargs)
        model_hidden_size = _resolve_hidden_size(model)
        checks["model_hidden_size"] = model_hidden_size
        checks["model_type"] = getattr(model.config, "model_type", "unknown")
        if getattr(model.config, "eos_token_id", None) is None:
            warnings.append("model.config.eos_token_id is missing")
        del model
    except Exception as exc:
        errors.append(f"failed to load model/tokenizer: {exc}")

    prompt_validation = validate_prompt_configuration(config.prompt, model_hidden_size=model_hidden_size)
    errors.extend(prompt_validation["errors"])
    warnings.extend(prompt_validation["warnings"])
    checks["prompt_validation"] = prompt_validation["details"]

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _evaluate(prompt_model, eval_loader, tokenizer, device, max_new_tokens, temperature, do_sample, top_k):
    prompt_model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions: list[str] = []
    references: list[str] = []

    for batch in tqdm(eval_loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        source_ids = batch["source_ids"].to(device)
        source_attention_mask = batch["source_attention_mask"].to(device)

        outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += float(outputs.loss.detach().cpu())
        total_batches += 1

        generated = prompt_model.generate(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend([_normalize_text(item) for item in decoded])
        references.extend([_normalize_text(item) for item in batch["target_texts"]])

    average_loss = total_loss / max(total_batches, 1)
    exact_match = 0.0
    if references:
        exact_match = sum(int(pred == ref) for pred, ref in zip(predictions, references)) / len(references)
    return {
        "loss": average_loss,
        "exact_match": exact_match,
        "predictions": predictions,
        "references": references,
    }


def _is_better_checkpoint(
    current_exact_match: float,
    current_eval_loss: float,
    best_exact_match: float | None,
    best_eval_loss: float | None,
) -> bool:
    if best_exact_match is None:
        return True
    if current_exact_match > best_exact_match:
        return True
    if current_exact_match < best_exact_match:
        return False
    if best_eval_loss is None:
        return True
    return current_eval_loss < best_eval_loss


def _build_prompt_model(config: ExperimentConfig, device: torch.device):
    model_name = config.model.name_or_path
    tokenizer_name = config.model.tokenizer_name_or_path
    prompt_model, tokenizer = SoftPromptCausalLM.from_pretrained(
        model_name_or_path=model_name,
        num_virtual_tokens=config.prompt.num_virtual_tokens,
        init_strategy=config.prompt.init_strategy,
        prompt_path=config.prompt.prompt_path,
        emotion_vector_route=config.prompt.emotion_vector_route,
        emotion_vectors_path=config.prompt.emotion_vectors_path,
        emotion_vector_metadata_path=config.prompt.emotion_vector_metadata_path,
        emotion_vector_projection_path=config.prompt.emotion_vector_projection_path,
        emotion_names=config.prompt.emotion_names,
        emotion_vector_combination=config.prompt.emotion_vector_combination,
        emotion_vector_l2_normalize=config.prompt.emotion_vector_l2_normalize,
        random_range=config.prompt.random_range,
        sampled_vocab_size=config.prompt.sampled_vocab_size,
        tokenizer_name_or_path=tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=config.model.torch_dtype,
    )
    prompt_model.to(device)
    return prompt_model, tokenizer


def evaluate_prompt_model(config: ExperimentConfig, prompt_path: str | Path | None = None) -> dict:
    set_seed(config.training.seed)
    device = resolve_device(config.model.device)
    prompt_model, tokenizer = _build_prompt_model(config, device)
    if prompt_path is not None:
        prompt_model.load_prompt(prompt_path)

    eval_dataset = load_jsonl_dataset(config.dataset, tokenizer, split="eval")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        collate_fn=CausalCollator(tokenizer),
    )
    metrics = _evaluate(
        prompt_model=prompt_model,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=config.training.max_new_tokens,
        temperature=config.training.temperature,
        do_sample=config.training.do_sample,
        top_k=config.training.top_k,
    )
    return {"metrics": {"loss": metrics["loss"], "exact_match": metrics["exact_match"]}, "predictions": metrics["predictions"]}


def train_prompt_model(config: ExperimentConfig) -> dict:
    set_seed(config.training.seed)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config_to_dict(config))

    device = resolve_device(config.model.device)
    prompt_model, tokenizer = _build_prompt_model(config, device)

    train_dataset = load_jsonl_dataset(config.dataset, tokenizer, split="train")
    eval_dataset = load_jsonl_dataset(config.dataset, tokenizer, split="eval")
    collator = CausalCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW(
        prompt_model.trainable_parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_exact_match: float | None = None
    best_eval_loss: float | None = None
    best_prompt_path = output_dir / "best_prompt.pt"
    history: list[dict[str, float]] = []
    step = 0
    train_iterator = iter(train_loader)

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

        outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.trainable_parameters(), config.training.gradient_clip_norm)
        optimizer.step()
        step += 1

        if step % config.training.logging_steps == 0 or step == 1:
            history.append({"step": step, "train_loss": float(loss.detach().cpu())})

        if step % config.training.eval_steps == 0 or step == config.training.max_steps:
            metrics = _evaluate(
                prompt_model=prompt_model,
                eval_loader=eval_loader,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=config.training.max_new_tokens,
                temperature=config.training.temperature,
                do_sample=config.training.do_sample,
                top_k=config.training.top_k,
            )
            current_metric = float(metrics["exact_match"])
            current_eval_loss = float(metrics["loss"])
            history.append(
                {
                    "step": step,
                    "eval_loss": current_eval_loss,
                    "eval_exact_match": current_metric,
                }
            )
            if _is_better_checkpoint(current_metric, current_eval_loss, best_exact_match, best_eval_loss):
                best_exact_match = current_metric
                best_eval_loss = current_eval_loss
                prompt_model.save_prompt(
                    best_prompt_path,
                    metadata={
                        "step": step,
                        "eval_loss": current_eval_loss,
                        "eval_exact_match": current_metric,
                    },
                )
                if config.output.save_predictions:
                    _write_json(
                        output_dir / "best_predictions.json",
                        {"predictions": metrics["predictions"], "references": metrics["references"]},
                    )

        if step % config.training.save_steps == 0:
            prompt_model.save_prompt(output_dir / f"prompt_step_{step}.pt", metadata={"step": step})

    result = evaluate_prompt_model(config, prompt_path=best_prompt_path)
    summary = {
        "best_prompt_path": str(best_prompt_path),
        "metrics": result["metrics"],
        "history": history,
    }
    if config.output.save_metrics:
        _write_json(output_dir / "metrics.json", summary)
    return summary