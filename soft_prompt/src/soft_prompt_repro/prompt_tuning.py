from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class SoftPromptT5(nn.Module):
    def __init__(
        self,
        model,
        num_virtual_tokens: int,
        init_strategy: str = "random_uniform",
        random_range: float = 0.5,
        sampled_vocab_size: int = 5000,
        tokenizer=None,
        label_texts: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.init_strategy = init_strategy
        self.random_range = random_range
        self.sampled_vocab_size = sampled_vocab_size
        self.hidden_size = int(model.config.d_model)
        self.prompt_embeddings = nn.Parameter(torch.empty(num_virtual_tokens, self.hidden_size))

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._initialize_prompt(tokenizer=tokenizer, label_texts=label_texts)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_virtual_tokens: int,
        init_strategy: str,
        random_range: float,
        sampled_vocab_size: int,
        label_texts: list[str] | None = None,
    ) -> tuple["SoftPromptT5", Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        return (
            cls(
                model=model,
                num_virtual_tokens=num_virtual_tokens,
                init_strategy=init_strategy,
                random_range=random_range,
                sampled_vocab_size=sampled_vocab_size,
                tokenizer=tokenizer,
                label_texts=label_texts,
            ),
            tokenizer,
        )

    def _initialize_prompt(self, tokenizer=None, label_texts: list[str] | None = None) -> None:
        embedding_table = self.model.get_input_embeddings().weight.detach()
        device = embedding_table.device
        if self.init_strategy == "random_uniform":
            nn.init.uniform_(self.prompt_embeddings, -self.random_range, self.random_range)
            return

        prompt_vectors: list[torch.Tensor] = []
        if self.init_strategy == "class_labels" and tokenizer is not None and label_texts:
            for label in label_texts:
                token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                prompt_vectors.append(embedding_table[token_ids].mean(dim=0))

        candidate_size = min(int(self.sampled_vocab_size), embedding_table.shape[0])
        sampled_ids = torch.arange(candidate_size, device=device)
        sampled_vectors = embedding_table[sampled_ids]
        cursor = 0
        while len(prompt_vectors) < self.num_virtual_tokens:
            prompt_vectors.append(sampled_vectors[cursor % sampled_vectors.shape[0]])
            cursor += 1

        self.prompt_embeddings.data.copy_(torch.stack(prompt_vectors[: self.num_virtual_tokens]))

    def build_prompted_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(input_embeddings.shape[0], -1, -1)
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
        return self.model(inputs_embeds=inputs_embeds, attention_mask=prompted_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generation_kwargs) -> torch.Tensor:
        inputs_embeds, prompted_mask = self.build_prompted_inputs(input_ids, attention_mask)
        encoder_outputs = self.model.get_encoder()(inputs_embeds=inputs_embeds, attention_mask=prompted_mask, return_dict=True)
        return self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=prompted_mask,
            **generation_kwargs,
        )

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
        if prompt.shape != self.prompt_embeddings.shape:
            raise ValueError(f"Prompt shape mismatch: expected {tuple(self.prompt_embeddings.shape)}, got {tuple(prompt.shape)}")
        self.prompt_embeddings.data.copy_(prompt)
        return payload.get("metadata", {})