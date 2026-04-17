from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts_from_jsonl(
    paths: list[str],
    *,
    text_field: str | None = None,
    input_field: str = "input",
    target_field: str = "target",
    join_fields: bool = False,
    max_examples: int | None = None,
) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if text_field:
                    text = str(row[text_field])
                elif join_fields:
                    text = f"{row[input_field]}\n{row[target_field]}"
                else:
                    text = str(row[target_field])
                texts.append(text)
                if max_examples is not None and len(texts) >= max_examples:
                    return texts
    return texts


def fit_linear_projection(source_matrix: np.ndarray, target_matrix: np.ndarray, ridge_alpha: float = 1e-4) -> np.ndarray:
    if source_matrix.ndim != 2 or target_matrix.ndim != 2:
        raise ValueError("source_matrix and target_matrix must both be 2D")
    if source_matrix.shape[0] != target_matrix.shape[0]:
        raise ValueError("source_matrix and target_matrix must have the same number of rows")

    x = source_matrix.astype(np.float32, copy=False)
    y = target_matrix.astype(np.float32, copy=False)
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(ridge_alpha)
    xty = x.T @ y
    projection = np.linalg.solve(xtx + reg, xty)
    return projection.astype(np.float32, copy=False)


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(torch_dtype: str | None):
    if not torch_dtype:
        return None
    return getattr(torch, torch_dtype)


def _load_model_and_tokenizer(model_name_or_path: str, trust_remote_code: bool, torch_dtype: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    resolved_dtype = _resolve_dtype(torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    return model, tokenizer


def _pool_valid_positions(hidden: torch.Tensor, attention_mask: torch.Tensor, token_pool_start: int) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    batch_size = hidden.shape[0]
    for index in range(batch_size):
        valid_positions = torch.nonzero(attention_mask[index].bool(), as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            pooled.append(hidden[index].mean(dim=0))
            continue
        pooled_positions = valid_positions[valid_positions >= int(token_pool_start)]
        if pooled_positions.numel() == 0:
            pooled_positions = valid_positions
        pooled.append(hidden[index, pooled_positions].mean(dim=0))
    return torch.stack(pooled, dim=0)


def _extract_representation(
    model,
    tokenizer,
    texts: list[str],
    *,
    representation: str,
    layer_idx: int,
    token_pool_start: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    model.to(device)
    model.eval()
    rows: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"extract:{representation}", leave=False):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            if representation == "embeddings":
                hidden = model.get_input_embeddings()(encoded["input_ids"])
            else:
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states
                if layer_idx >= 0:
                    state_index = layer_idx + 1
                else:
                    state_index = layer_idx
                hidden = hidden_states[state_index]
        pooled = _pool_valid_positions(hidden, encoded["attention_mask"], token_pool_start)
        rows.append(pooled.detach().float().cpu().numpy())

    return np.concatenate(rows, axis=0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a linear projection from source model vectors to target model space")
    parser.add_argument("--texts-file", dest="texts_files", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--join-fields", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)

    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-representation", choices=["hidden", "embeddings"], default="hidden")
    parser.add_argument("--source-layer-idx", type=int, default=26)
    parser.add_argument("--source-token-pool-start", type=int, default=50)
    parser.add_argument("--source-trust-remote-code", action="store_true")
    parser.add_argument("--source-torch-dtype", default=None)

    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-representation", choices=["hidden", "embeddings"], default="embeddings")
    parser.add_argument("--target-layer-idx", type=int, default=-1)
    parser.add_argument("--target-token-pool-start", type=int, default=0)
    parser.add_argument("--target-trust-remote-code", action="store_true")
    parser.add_argument("--target-torch-dtype", default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    texts = load_texts_from_jsonl(
        args.texts_files,
        text_field=args.text_field,
        input_field=args.input_field,
        target_field=args.target_field,
        join_fields=args.join_fields,
        max_examples=args.max_examples,
    )
    if not texts:
        raise ValueError("No texts were loaded for projection fitting")

    device = _resolve_device(args.device)

    source_model, source_tokenizer = _load_model_and_tokenizer(
        args.source_model,
        trust_remote_code=args.source_trust_remote_code,
        torch_dtype=args.source_torch_dtype,
    )
    target_model, target_tokenizer = _load_model_and_tokenizer(
        args.target_model,
        trust_remote_code=args.target_trust_remote_code,
        torch_dtype=args.target_torch_dtype,
    )

    source_matrix = _extract_representation(
        source_model,
        source_tokenizer,
        texts,
        representation=args.source_representation,
        layer_idx=args.source_layer_idx,
        token_pool_start=args.source_token_pool_start,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    target_matrix = _extract_representation(
        target_model,
        target_tokenizer,
        texts,
        representation=args.target_representation,
        layer_idx=args.target_layer_idx,
        token_pool_start=args.target_token_pool_start,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    projection = fit_linear_projection(source_matrix, target_matrix, ridge_alpha=args.ridge_alpha)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, projection)

    metadata = {
        "texts_files": args.texts_files,
        "example_count": len(texts),
        "source_model": args.source_model,
        "source_representation": args.source_representation,
        "source_layer_idx": args.source_layer_idx,
        "source_token_pool_start": args.source_token_pool_start,
        "target_model": args.target_model,
        "target_representation": args.target_representation,
        "target_layer_idx": args.target_layer_idx,
        "target_token_pool_start": args.target_token_pool_start,
        "ridge_alpha": args.ridge_alpha,
        "projection_shape": list(projection.shape),
    }
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "metadata_path": str(metadata_path), "projection_shape": list(projection.shape)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()