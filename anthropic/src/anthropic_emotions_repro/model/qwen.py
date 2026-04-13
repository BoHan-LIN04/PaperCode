from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from anthropic_emotions_repro.utils.torch_utils import parse_attn_impl, parse_dtype


def resolve_pretrained_source(model_name: str) -> str:
    try:
        local_path = snapshot_download(repo_id=model_name, local_files_only=True)
        if local_path and Path(local_path).exists():
            return local_path
    except Exception:
        pass
    return model_name


def load_causal_lm_with_fallback(
    model_name: str,
    *,
    dtype: torch.dtype,
    attn_impl: str | None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
):
    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl
    source = resolve_pretrained_source(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        return model, attn_impl
    except ImportError as exc:
        msg = str(exc).lower()
        if attn_impl == "flash_attention_2" and ("flash_attn" in msg or "flashattention2" in msg):
            fallback = "sdpa" if torch.cuda.is_available() else "eager"
            retry_kwargs = dict(kwargs)
            retry_kwargs["attn_implementation"] = fallback
            model = AutoModelForCausalLM.from_pretrained(source, **retry_kwargs)
            return model, fallback
        raise


def load_tokenizer_with_fallback(model_name: str, *, trust_remote_code: bool = True):
    source = resolve_pretrained_source(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        return tokenizer, source
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=False,
        )
        return tokenizer, model_name


class QwenHookedModel:
    def __init__(
        self,
        model_name: str,
        layer_idx: int,
        dtype: str = "bfloat16",
        attn_impl: str | None = "flash_attention_2",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        use_cache: bool = False,
    ) -> None:
        self.model_name = model_name
        self.layer_idx = int(layer_idx)
        self.dtype = parse_dtype(dtype)
        self.attn_impl = parse_attn_impl(attn_impl)

        self.tokenizer, tokenizer_source = load_tokenizer_with_fallback(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        local_only = tokenizer_source != model_name or Path(model_name).exists()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model, self.attn_impl = load_causal_lm_with_fallback(
            model_name=model_name,
            dtype=self.dtype,
            attn_impl=self.attn_impl,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_only,
        )
        self.model.config.use_cache = use_cache
        self.model.eval()

        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = len(self.model.model.layers)
        if self.layer_idx < 0:
            self.layer_idx = self.num_layers + self.layer_idx
        if not 0 <= self.layer_idx < self.num_layers:
            raise ValueError(f"layer_idx out of range: {self.layer_idx} / {self.num_layers}")

    @property
    def hooked_layer(self):
        return self.model.model.layers[self.layer_idx]

    @contextmanager
    def capture_residual(self) -> Iterator[dict[str, torch.Tensor]]:
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden
            return output

        handle = self.hooked_layer.register_forward_hook(_hook)
        try:
            yield captured
        finally:
            handle.remove()

    def encode_batch(self, texts: list[str], max_length: int, device: str | None = None) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        target_device = device or str(self.model.device)
        return {k: v.to(target_device) for k, v in encoded.items()}

    @torch.inference_mode()
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with self.capture_residual() as bucket:
            _ = self.model(**inputs)
        return bucket["hidden"]

    @torch.inference_mode()
    def generate(self, prompt: str, *, generation_defaults: dict) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(generation_defaults.get("max_new_tokens", 192)),
            temperature=float(generation_defaults.get("temperature", 0.7)),
            top_p=float(generation_defaults.get("top_p", 0.8)),
            top_k=int(generation_defaults.get("top_k", 20)),
            do_sample=bool(generation_defaults.get("do_sample", True)),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
