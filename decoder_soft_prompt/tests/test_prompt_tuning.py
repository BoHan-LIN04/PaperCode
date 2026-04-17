from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from decoder_soft_prompt_repro.prompt_tuning import SoftPromptCausalLM


class _DummyEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size, "eos_token_id": 1})()
        self.embeddings = _DummyEmbeddings(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None, labels=None):
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return type("Output", (), {"logits": logits, "loss": loss})()


def test_build_prompted_inputs_adds_virtual_tokens():
    model = _DummyModel()
    prompt_model = SoftPromptCausalLM(model=model, num_virtual_tokens=3)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    prompted_embeddings, prompted_mask = prompt_model.build_prompted_inputs(input_ids, attention_mask)

    assert prompted_embeddings.shape == (1, 6, 8)
    assert prompted_mask.tolist() == [[1, 1, 1, 1, 1, 1]]


def test_forward_prepends_ignore_labels_for_prompt_tokens():
    model = _DummyModel()
    prompt_model = SoftPromptCausalLM(model=model, num_virtual_tokens=2)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[-100, 2, 3]], dtype=torch.long)

    outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None


def test_build_prompted_inputs_matches_model_embedding_dtype():
    model = _DummyModel(hidden_size=8)
    model.embeddings.weight.data = model.embeddings.weight.data.to(torch.bfloat16)
    model.lm_head.weight.data = model.lm_head.weight.data.to(torch.bfloat16)
    prompt_model = SoftPromptCausalLM(model=model, num_virtual_tokens=2)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    prompted_embeddings, _ = prompt_model.build_prompted_inputs(input_ids, attention_mask)

    assert prompted_embeddings.dtype == torch.bfloat16


def test_emotion_vector_initialization_selects_named_rows_and_repeats(tmp_path: Path):
    model = _DummyModel(hidden_size=8)
    vectors = np.asarray(
        [
            [1.0] * 8,
            [2.0] * 8,
            [3.0] * 8,
        ],
        dtype=np.float32,
    )
    vectors_path = tmp_path / "emotion_vectors_orth.npy"
    metadata_path = tmp_path / "vector_metadata.json"
    np.save(vectors_path, vectors)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful", "calm", "hopeful"]}), encoding="utf-8")

    prompt_model = SoftPromptCausalLM(
        model=model,
        num_virtual_tokens=5,
        init_strategy="emotion_vectors",
        emotion_vector_route="same_model",
        emotion_vectors_path=str(vectors_path),
        emotion_vector_metadata_path=str(metadata_path),
        emotion_names=["hopeful", "joyful"],
        emotion_vector_combination="repeat",
    )

    expected = torch.tensor(
        [
            [3.0] * 8,
            [1.0] * 8,
            [3.0] * 8,
            [1.0] * 8,
            [3.0] * 8,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(prompt_model.prompt_embeddings.detach().cpu(), expected)


def test_emotion_vector_initialization_uses_projection_when_hidden_sizes_differ(tmp_path: Path):
    model = _DummyModel(hidden_size=4)
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    projection = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    vectors_path = tmp_path / "emotion_vectors_orth.npy"
    metadata_path = tmp_path / "vector_metadata.json"
    projection_path = tmp_path / "projection.npy"
    np.save(vectors_path, vectors)
    np.save(projection_path, projection)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful", "calm"]}), encoding="utf-8")

    prompt_model = SoftPromptCausalLM(
        model=model,
        num_virtual_tokens=2,
        init_strategy="emotion_vectors",
        emotion_vector_route="projected",
        emotion_vectors_path=str(vectors_path),
        emotion_vector_metadata_path=str(metadata_path),
        emotion_vector_projection_path=str(projection_path),
        emotion_names=["calm", "joyful"],
        emotion_vector_combination="repeat",
    )

    expected = torch.tensor(
        [
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(prompt_model.prompt_embeddings.detach().cpu(), expected)


def test_same_model_route_rejects_projection(tmp_path: Path):
    model = _DummyModel(hidden_size=3)
    vectors = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    projection = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    vectors_path = tmp_path / "emotion_vectors_orth.npy"
    metadata_path = tmp_path / "vector_metadata.json"
    projection_path = tmp_path / "projection.npy"
    np.save(vectors_path, vectors)
    np.save(projection_path, projection)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful"]}), encoding="utf-8")

    try:
        SoftPromptCausalLM(
            model=model,
            num_virtual_tokens=1,
            init_strategy="emotion_vectors",
            emotion_vector_route="same_model",
            emotion_vectors_path=str(vectors_path),
            emotion_vector_metadata_path=str(metadata_path),
            emotion_vector_projection_path=str(projection_path),
            emotion_names=["joyful"],
        )
    except ValueError as exc:
        assert "same_model" in str(exc)
    else:
        raise AssertionError("Expected same_model route to reject projection path")


def test_projected_route_requires_projection(tmp_path: Path):
    model = _DummyModel(hidden_size=4)
    vectors = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    vectors_path = tmp_path / "emotion_vectors_orth.npy"
    metadata_path = tmp_path / "vector_metadata.json"
    np.save(vectors_path, vectors)
    metadata_path.write_text(json.dumps({"emotion_names": ["joyful"]}), encoding="utf-8")

    try:
        SoftPromptCausalLM(
            model=model,
            num_virtual_tokens=1,
            init_strategy="emotion_vectors",
            emotion_vector_route="projected",
            emotion_vectors_path=str(vectors_path),
            emotion_vector_metadata_path=str(metadata_path),
            emotion_names=["joyful"],
        )
    except ValueError as exc:
        assert "requires emotion_vector_projection_path" in str(exc)
    else:
        raise AssertionError("Expected projected route to require a projection path")