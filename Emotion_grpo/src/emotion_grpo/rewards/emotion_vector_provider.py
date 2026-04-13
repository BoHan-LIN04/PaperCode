from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from emotion_grpo.rewards.base import IntrinsicRewardProvider


def canonicalize_emotion_name(name: Any, aliases: dict[str, str] | None = None) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return ""
    if aliases and text in aliases:
        return aliases[text].strip().lower()
    return text


def parse_layer_idx_from_artifact_root(path: str | Path) -> int | None:
    match = re.search(r"__layer=(\d+)", str(path))
    if match is None:
        return None
    return int(match.group(1))


def resolve_artifact_paths(
    *,
    artifact_root: str | Path | None,
    vector_path: str | Path | None,
    vector_metadata_path: str | Path | None,
    neutral_embeddings_path: str | Path | None,
) -> tuple[Path, Path, Path | None]:
    if vector_path is not None and vector_metadata_path is not None:
        return Path(vector_path), Path(vector_metadata_path), Path(neutral_embeddings_path) if neutral_embeddings_path else None
    if artifact_root is None:
        raise ValueError("Either artifact_root or both vector_path and vector_metadata_path must be provided")

    root = Path(artifact_root)
    base = root / "05_emotion_vectors" / "intermediate"
    resolved_vector = Path(vector_path) if vector_path is not None else base / "emotion_vectors_orth.npy"
    resolved_meta = Path(vector_metadata_path) if vector_metadata_path is not None else base / "vector_metadata.json"
    resolved_neutral = (
        Path(neutral_embeddings_path)
        if neutral_embeddings_path is not None
        else root / "04_activation_cache" / "intermediate" / "neutral_story_embeddings.npy"
    )
    return resolved_vector, resolved_meta, resolved_neutral


def compute_neutral_components(
    neutral_embeddings: np.ndarray,
    *,
    pc_count: int | None = None,
    variance_target: float | None = None,
) -> np.ndarray:
    if neutral_embeddings.ndim != 2:
        raise ValueError(f"neutral_embeddings must be 2D, got shape {neutral_embeddings.shape}")
    centered = neutral_embeddings.astype(np.float32, copy=False) - neutral_embeddings.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((0, centered.shape[1]), dtype=np.float32)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((0, centered.shape[1]), dtype=np.float32)
    if pc_count is None:
        ratios = singular_values**2
        ratios = ratios / np.clip(ratios.sum(), 1e-8, None)
        target = float(variance_target if variance_target is not None else 0.5)
        cumulative = np.cumsum(ratios)
        pc_count = int(np.searchsorted(cumulative, target) + 1)
    pc_count = max(0, min(int(pc_count), vh.shape[0]))
    if pc_count == 0:
        return np.zeros((0, centered.shape[1]), dtype=np.float32)
    return vh[:pc_count].astype(np.float32, copy=False)


def project_out_components(embedding: np.ndarray, components: np.ndarray) -> np.ndarray:
    if components.size == 0:
        return embedding.astype(np.float32, copy=False)
    projected = embedding @ components.T @ components
    return embedding.astype(np.float32, copy=False) - projected.astype(np.float32, copy=False)


def score_embedding_against_vectors(
    embedding: np.ndarray,
    vectors: np.ndarray,
    *,
    target_index: int,
    score_mode: str = "margin",
    reward_scale: float = 1.0,
    reward_clip: float | None = 1.0,
) -> float:
    emb = embedding.astype(np.float32, copy=False)
    vecs = vectors.astype(np.float32, copy=False)
    emb_norm = float(np.linalg.norm(emb))
    vec_norms = np.linalg.norm(vecs, axis=1)
    denom = np.clip(vec_norms * max(emb_norm, 1e-8), 1e-8, None)
    cosine_scores = (vecs @ emb) / denom

    if score_mode == "cosine":
        raw_score = float(cosine_scores[target_index])
    elif score_mode == "dot":
        raw_score = float(vecs[target_index] @ emb)
    elif score_mode == "margin":
        target_score = float(cosine_scores[target_index])
        if cosine_scores.shape[0] == 1:
            raw_score = target_score
        else:
            other_scores = np.delete(cosine_scores, target_index)
            raw_score = target_score - float(other_scores.max())
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    scaled = raw_score * float(reward_scale)
    if reward_clip is None:
        return float(scaled)
    clip_value = abs(float(reward_clip))
    return float(np.clip(scaled, -clip_value, clip_value))


class _ResidualScorer:
    def __init__(
        self,
        *,
        model_name: str,
        layer_idx: int,
        device: str = "auto",
        dtype: str | None = None,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        max_length: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.layer_idx = int(layer_idx)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = None
        if dtype:
            torch_dtype = getattr(torch, dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.device = resolved_device
        self.model.to(self.device)
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

    def encode(self, texts: list[str]) -> dict[str, Any]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def hidden_states(self, texts: list[str]) -> tuple[Any, Any]:
        torch = self._torch
        inputs = self.encode(texts)
        bucket: dict[str, Any] = {}

        def _hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            bucket["hidden"] = hidden
            return output

        handle = self.hooked_layer.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                _ = self.model(**inputs)
        finally:
            handle.remove()
        return bucket["hidden"], inputs["attention_mask"]


class EmotionVectorRewardProvider(IntrinsicRewardProvider):
    def __init__(
        self,
        *,
        artifact_root: str | None = None,
        vector_path: str | None = None,
        vector_metadata_path: str | None = None,
        neutral_embeddings_path: str | None = None,
        score_model_name: str = "Qwen/Qwen3-0.6B",
        layer_idx: int | None = None,
        target_field: str = "ground_truth",
        target_aliases: dict[str, str] | None = None,
        score_mode: str = "margin",
        reward_scale: float = 2.0,
        reward_clip: float | None = 1.0,
        empty_response_penalty: float = -1.0,
        missing_target_penalty: float = -1.0,
        unknown_target_penalty: float = -1.0,
        use_neutral_pca: bool = True,
        neutral_pc_count: int | None = None,
        neutral_pca_variance: float | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        local_files_only: bool = False,
        max_length: int = 256,
        trust_remote_code: bool = True,
    ) -> None:
        self.target_field = target_field
        self.target_aliases = {str(k).lower(): str(v).lower() for k, v in (target_aliases or {}).items()}
        self.score_mode = score_mode
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.empty_response_penalty = float(empty_response_penalty)
        self.missing_target_penalty = float(missing_target_penalty)
        self.unknown_target_penalty = float(unknown_target_penalty)

        resolved_vector_path, resolved_meta_path, resolved_neutral_path = resolve_artifact_paths(
            artifact_root=artifact_root,
            vector_path=vector_path,
            vector_metadata_path=vector_metadata_path,
            neutral_embeddings_path=neutral_embeddings_path,
        )

        self.vector_path = resolved_vector_path
        self.vector_metadata_path = resolved_meta_path
        self.neutral_embeddings_path = resolved_neutral_path

        with self.vector_metadata_path.open("r", encoding="utf-8") as handle:
            self.vector_metadata = json.load(handle)
        self.vectors = np.load(self.vector_path).astype(np.float32, copy=False)
        self.emotion_names = [canonicalize_emotion_name(name) for name in self.vector_metadata["emotion_names"]]
        self.emotion_to_index = {name: idx for idx, name in enumerate(self.emotion_names)}
        self.token_pool_start = int(self.vector_metadata.get("token_pool_start", 0))

        neutral_components = np.zeros((0, self.vectors.shape[1]), dtype=np.float32)
        if use_neutral_pca and self.neutral_embeddings_path is not None and self.neutral_embeddings_path.exists():
            neutral_embeddings = np.load(self.neutral_embeddings_path).astype(np.float32, copy=False)
            pc_count = neutral_pc_count
            if pc_count is None:
                pc_count = self.vector_metadata.get("pc_count_removed")
            neutral_components = compute_neutral_components(
                neutral_embeddings,
                pc_count=pc_count,
                variance_target=neutral_pca_variance if neutral_pca_variance is not None else self.vector_metadata.get("variance_target"),
            )
        self.neutral_components = neutral_components

        inferred_layer_idx = parse_layer_idx_from_artifact_root(artifact_root) if artifact_root else None
        self.layer_idx = int(layer_idx if layer_idx is not None else inferred_layer_idx if inferred_layer_idx is not None else 0)
        self.scorer = _ResidualScorer(
            model_name=score_model_name,
            layer_idx=self.layer_idx,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            max_length=max_length,
        )
        if self.scorer.hidden_size != self.vectors.shape[1]:
            raise ValueError(
                "Hidden-size mismatch between scorer model and emotion vectors: "
                f"{self.scorer.hidden_size} != {self.vectors.shape[1]}"
            )

    def _resolve_target_name(self, meta: dict[str, Any]) -> str:
        candidate = meta.get(self.target_field)
        if candidate in (None, "") and self.target_field != "ground_truth":
            candidate = meta.get("ground_truth", meta.get("label", ""))
        elif candidate in (None, ""):
            candidate = meta.get("label", "")
        return canonicalize_emotion_name(candidate, aliases=self.target_aliases)

    def _pool_embedding(self, hidden_row: np.ndarray, attention_row: np.ndarray) -> np.ndarray:
        valid_positions = np.nonzero(attention_row.astype(bool))[0]
        if valid_positions.size == 0:
            return hidden_row.mean(axis=0).astype(np.float32, copy=False)
        pooled_positions = valid_positions[valid_positions >= self.token_pool_start]
        if pooled_positions.size == 0:
            pooled_positions = valid_positions
        return hidden_row[pooled_positions].mean(axis=0).astype(np.float32, copy=False)

    def _score_generation(self, generation: str, target_name: str) -> float:
        if not generation.strip():
            return self.empty_response_penalty
        if not target_name:
            return self.missing_target_penalty
        if target_name not in self.emotion_to_index:
            return self.unknown_target_penalty

        hidden, attention = self.scorer.hidden_states([generation])
        embedding = self._pool_embedding(
            hidden[0].detach().float().cpu().numpy(),
            attention[0].detach().cpu().numpy(),
        )
        embedding = project_out_components(embedding, self.neutral_components)
        return score_embedding_against_vectors(
            embedding,
            self.vectors,
            target_index=self.emotion_to_index[target_name],
            score_mode=self.score_mode,
            reward_scale=self.reward_scale,
            reward_clip=self.reward_clip,
        )

    def score_batch(
        self,
        batch_records: list[dict[str, Any]],
        generations: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[float]:
        if not (len(batch_records) == len(generations) == len(metadata)):
            raise ValueError("batch_records, generations, and metadata must have the same length")
        del batch_records
        return [
            self._score_generation(generation=generation, target_name=self._resolve_target_name(meta))
            for generation, meta in zip(generations, metadata)
        ]
