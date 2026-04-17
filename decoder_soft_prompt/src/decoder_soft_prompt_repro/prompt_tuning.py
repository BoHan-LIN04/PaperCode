from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_hidden_size(model) -> int:
    for attribute in ["hidden_size", "n_embd", "d_model"]:
        value = getattr(model.config, attribute, None)
        if value is not None:
            return int(value)
    embedding = model.get_input_embeddings()
    return int(embedding.weight.shape[1])


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tensor_payload(path: str | Path) -> torch.Tensor:
    source = Path(path)
    if source.suffix == ".npy":
        return torch.from_numpy(np.load(source)).float()
    payload = torch.load(source, map_location="cpu")
    if isinstance(payload, dict):
        for key in ["prompt_embeddings", "projection", "tensor", "weight"]:
            if key in payload:
                return payload[key].float()
    if isinstance(payload, torch.Tensor):
        return payload.float()
    raise ValueError(f"Unsupported tensor payload in {path}")


def _l2_normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    denom = torch.linalg.norm(matrix, dim=1, keepdim=True)
    denom = torch.clamp(denom, min=1e-8)
    return matrix / denom


def _expand_emotion_prompt(selected_vectors: torch.Tensor, prompt_length: int, combination: str) -> torch.Tensor:
    if combination == "mean_then_repeat":
        row = selected_vectors.mean(dim=0, keepdim=True)
        return row.repeat(prompt_length, 1)

    if combination == "repeat":
        repeat_count = int(np.ceil(prompt_length / selected_vectors.shape[0]))
        tiled = selected_vectors.repeat(repeat_count, 1)
        return tiled[:prompt_length]

    if combination == "interleave":
        rows = [selected_vectors[index % selected_vectors.shape[0]] for index in range(prompt_length)]
        return torch.stack(rows, dim=0)

    raise ValueError(f"Unsupported emotion_vector_combination: {combination}")


class SoftPromptCausalLM(nn.Module):
    def __init__(
        self,
        model,
        num_virtual_tokens: int,
        init_strategy: str = "random_uniform",
        prompt_path: str | None = None,
        emotion_vector_route: str = "same_model",
        emotion_vectors_path: str | None = None,
        emotion_vector_metadata_path: str | None = None,
        emotion_vector_projection_path: str | None = None,
        emotion_names: list[str] | None = None,
        emotion_vector_combination: str = "repeat",
        emotion_vector_l2_normalize: bool = False,
        random_range: float = 0.5,
        sampled_vocab_size: int = 5000,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.init_strategy = init_strategy
        self.prompt_path = prompt_path
        self.emotion_vector_route = str(emotion_vector_route).strip().lower()
        self.emotion_vectors_path = emotion_vectors_path
        self.emotion_vector_metadata_path = emotion_vector_metadata_path
        self.emotion_vector_projection_path = emotion_vector_projection_path
        self.emotion_names = [str(name).strip().lower() for name in (emotion_names or []) if str(name).strip()]
        self.emotion_vector_combination = emotion_vector_combination
        self.emotion_vector_l2_normalize = emotion_vector_l2_normalize
        self.random_range = random_range
        self.sampled_vocab_size = sampled_vocab_size
        self.hidden_size = _resolve_hidden_size(model)
        self.prompt_embeddings = nn.Parameter(torch.empty(num_virtual_tokens, self.hidden_size))

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._initialize_prompt()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_virtual_tokens: int,
        init_strategy: str = "random_uniform",
        prompt_path: str | None = None,
        emotion_vector_route: str = "same_model",
        emotion_vectors_path: str | None = None,
        emotion_vector_metadata_path: str | None = None,
        emotion_vector_projection_path: str | None = None,
        emotion_names: list[str] | None = None,
        emotion_vector_combination: str = "repeat",
        emotion_vector_l2_normalize: bool = False,
        random_range: float = 0.5,
        sampled_vocab_size: int = 5000,
        tokenizer_name_or_path: str | None = None,
        trust_remote_code: bool = True,
        torch_dtype: str | None = None,
    ) -> tuple["SoftPromptCausalLM", Any]:
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype:
            model_kwargs["dtype"] = getattr(torch, torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return (
            cls(
                model=model,
                num_virtual_tokens=num_virtual_tokens,
                init_strategy=init_strategy,
                prompt_path=prompt_path,
                emotion_vector_route=emotion_vector_route,
                emotion_vectors_path=emotion_vectors_path,
                emotion_vector_metadata_path=emotion_vector_metadata_path,
                emotion_vector_projection_path=emotion_vector_projection_path,
                emotion_names=emotion_names,
                emotion_vector_combination=emotion_vector_combination,
                emotion_vector_l2_normalize=emotion_vector_l2_normalize,
                random_range=random_range,
                sampled_vocab_size=sampled_vocab_size,
            ),
            tokenizer,
        )

    def _load_external_prompt(self, path: str) -> torch.Tensor:
        value = _load_tensor_payload(path)
        expected_shape = (self.num_virtual_tokens, self.hidden_size)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Prompt shape mismatch: expected {expected_shape}, got {tuple(value.shape)}")
        return value

    def _initialize_from_emotion_vectors(self, device: torch.device) -> None:
        if self.emotion_vector_route not in {"same_model", "projected"}:
            raise ValueError(
                "emotion_vector_route must be either 'same_model' or 'projected', got: "
                f"{self.emotion_vector_route}"
            )
        if not self.emotion_vectors_path:
            raise ValueError("emotion_vectors_path is required when init_strategy=emotion_vectors")
        if not self.emotion_vector_metadata_path:
            raise ValueError("emotion_vector_metadata_path is required when init_strategy=emotion_vectors")
        if not self.emotion_names:
            raise ValueError("emotion_names must contain at least one value when init_strategy=emotion_vectors")

        vectors = torch.from_numpy(np.load(self.emotion_vectors_path)).float()
        metadata = _load_json(self.emotion_vector_metadata_path)
        available_names = [str(name).strip().lower() for name in metadata["emotion_names"]]
        name_to_index = {name: index for index, name in enumerate(available_names)}

        missing = [name for name in self.emotion_names if name not in name_to_index]
        if missing:
            raise ValueError(f"Unknown emotion names: {missing}. Available names: {available_names}")

        selected = torch.stack([vectors[name_to_index[name]] for name in self.emotion_names], dim=0)
        if self.emotion_vector_l2_normalize:
            selected = _l2_normalize_rows(selected)

        if self.emotion_vector_route == "same_model":
            if self.emotion_vector_projection_path:
                raise ValueError(
                    "emotion_vector_route='same_model' does not allow emotion_vector_projection_path; "
                    "use route='projected' if you want to apply a projection"
                )
            if selected.shape[1] != self.hidden_size:
                raise ValueError(
                    "emotion_vector_route='same_model' requires matching hidden sizes: "
                    f"{selected.shape[1]} != {self.hidden_size}. Use route='projected' instead."
                )
        else:
            if not self.emotion_vector_projection_path:
                raise ValueError(
                    "emotion_vector_route='projected' requires emotion_vector_projection_path"
                )

        if self.emotion_vector_route == "projected":
            projection = _load_tensor_payload(self.emotion_vector_projection_path)
            expected_shape = (selected.shape[1], self.hidden_size)
            if tuple(projection.shape) != expected_shape:
                raise ValueError(
                    f"Projection shape mismatch: expected {expected_shape}, got {tuple(projection.shape)}"
                )
            selected = selected @ projection

        prompt = _expand_emotion_prompt(selected, self.num_virtual_tokens, self.emotion_vector_combination)
        self.prompt_embeddings.data.copy_(prompt.to(device))

    def _initialize_prompt(self) -> None:
        embedding_table = self.model.get_input_embeddings().weight.detach()
        device = embedding_table.device

        if self.init_strategy == "random_uniform":
            nn.init.uniform_(self.prompt_embeddings, -self.random_range, self.random_range)
            return

        if self.init_strategy == "from_file":
            if not self.prompt_path:
                raise ValueError("prompt_path is required when init_strategy=from_file")
            prompt = self._load_external_prompt(self.prompt_path).to(device)
            self.prompt_embeddings.data.copy_(prompt)
            return

        if self.init_strategy == "emotion_vectors":
            self._initialize_from_emotion_vectors(device)
            return

        if self.init_strategy == "sampled_vocab":
            candidate_size = min(int(self.sampled_vocab_size), embedding_table.shape[0])
            indices = torch.arange(candidate_size, device=device)
            samples = embedding_table[indices]
            if samples.shape[0] < self.num_virtual_tokens:
                repeat_count = int(np.ceil(self.num_virtual_tokens / samples.shape[0]))
                samples = samples.repeat(repeat_count, 1)
            self.prompt_embeddings.data.copy_(samples[: self.num_virtual_tokens])
            return

        raise ValueError(f"Unsupported init_strategy: {self.init_strategy}")

    def build_prompted_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt = self.prompt_embeddings.to(dtype=input_embeddings.dtype, device=input_embeddings.device)
        prompt = prompt.unsqueeze(0).expand(input_embeddings.shape[0], -1, -1)
        prompt_mask = torch.ones(
            input_embeddings.shape[0],
            self.num_virtual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        prompted_embeddings = torch.cat([prompt, input_embeddings], dim=1)
        prompted_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        return prompted_embeddings, prompted_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        inputs_embeds, prompted_mask = self.build_prompted_inputs(input_ids, attention_mask)
        prompted_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], self.num_virtual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            prompted_labels = torch.cat([prefix_labels, labels], dim=1)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=prompted_mask, labels=prompted_labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> torch.Tensor:
        generated = []
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        eos_token_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            inputs_embeds, prompted_mask = self.build_prompted_inputs(current_ids, current_mask)
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=prompted_mask)
            logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated.append(next_token)

            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            next_mask = torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device)
            current_mask = torch.cat([current_mask, next_mask], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if bool(torch.all(finished)):
                    break

        if not generated:
            return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
        return torch.stack(generated, dim=1)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.prompt_embeddings]

    def save_prompt(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_embeddings": self.prompt_embeddings.detach().cpu(),
            "num_virtual_tokens": self.num_virtual_tokens,
            "metadata": metadata or {},
        }
        torch.save(payload, destination)

    def load_prompt(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.prompt_embeddings.device)
        prompt = payload["prompt_embeddings"].to(self.prompt_embeddings.device)
        if tuple(prompt.shape) != tuple(self.prompt_embeddings.shape):
            raise ValueError(
                f"Prompt shape mismatch: expected {tuple(self.prompt_embeddings.shape)}, got {tuple(prompt.shape)}"
            )
        self.prompt_embeddings.data.copy_(prompt)
        return payload.get("metadata", {})