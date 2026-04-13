from __future__ import annotations

import torch


def parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": torch.bfloat16,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[key]


def parse_attn_impl(attn_impl: str | None) -> str | None:
    if not attn_impl:
        return None
    value = attn_impl.lower()
    if value in {"none", "null", "off"}:
        return None
    if value not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")
    return value
