from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from emotion_grpo.rewards.base import IntrinsicRewardProvider
from emotion_grpo.rewards.emotion_vector_provider import (
    _ResidualScorer,
    compute_neutral_components,
    project_out_components,
)


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _first_non_empty(row: dict[str, Any], field_names: list[str]) -> str:
    for name in field_names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _pool_embedding(hidden_row: np.ndarray, attention_row: np.ndarray, token_pool_start: int) -> np.ndarray:
    valid_positions = np.nonzero(attention_row.astype(bool))[0]
    if valid_positions.size == 0:
        return hidden_row.mean(axis=0).astype(np.float32, copy=False)
    pooled_positions = valid_positions[valid_positions >= int(token_pool_start)]
    if pooled_positions.size == 0:
        pooled_positions = valid_positions
    return hidden_row[pooled_positions].mean(axis=0).astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), 1e-8)
    return float(np.dot(a, b) / denom)


def _score_against_logic_centroids(
    embedding: np.ndarray,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
    *,
    score_mode: str = "margin",
    reward_scale: float = 1.0,
    reward_clip: float | None = 1.0,
) -> float:
    if score_mode == "margin":
        raw = _cosine(embedding, positive_centroid) - _cosine(embedding, negative_centroid)
    elif score_mode == "positive_cosine":
        raw = _cosine(embedding, positive_centroid)
    elif score_mode == "vector_cosine":
        raw = _cosine(embedding, positive_centroid - negative_centroid)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    scaled = raw * float(reward_scale)
    if reward_clip is None:
        return float(scaled)
    clip_value = abs(float(reward_clip))
    return float(np.clip(scaled, -clip_value, clip_value))


def load_reasoning_examples(
    source_files: list[str] | list[Path],
    *,
    text_fields: list[str] | None = None,
    label_field: str = "is_good",
) -> tuple[list[str], list[bool]]:
    fields = text_fields or ["thinking", "answer_content"]
    texts: list[str] = []
    labels: list[bool] = []
    for source_file in source_files:
        with Path(source_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                text = _first_non_empty(row, fields)
                if not text:
                    continue
                label = bool(row.get(label_field, False))
                texts.append(text)
                labels.append(label)
    return texts, labels


def build_logical_vector_artifact(
    *,
    source_files: list[str] | list[Path],
    artifact_dir: str | Path,
    score_model_name: str,
    layer_idx: int,
    token_pool_start: int = 8,
    batch_size: int = 4,
    text_fields: list[str] | None = None,
    label_field: str = "is_good",
    device: str = "cpu",
    dtype: str | None = None,
    local_files_only: bool = False,
    max_length: int = 768,
    use_neutral_pca: bool = True,
    neutral_pca_variance: float = 0.5,
    neutral_pc_count: int | None = None,
) -> dict[str, str]:
    texts, labels = load_reasoning_examples(source_files, text_fields=text_fields, label_field=label_field)
    if not texts:
        raise ValueError("No usable reasoning examples were found to build logical vectors")
    if not any(labels):
        raise ValueError("No positive examples were found in the logical vector sources")
    if all(labels):
        raise ValueError("No negative examples were found in the logical vector sources")

    scorer = _ResidualScorer(
        model_name=score_model_name,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        max_length=max_length,
    )
    artifact_root = _ensure_dir(artifact_dir)

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        hidden, attention = scorer.hidden_states(batch_texts)
        hidden_np = hidden.detach().float().cpu().numpy()
        attention_np = attention.detach().cpu().numpy()
        for hidden_row, attention_row in zip(hidden_np, attention_np):
            embeddings.append(_pool_embedding(hidden_row, attention_row, token_pool_start))

    emb_array = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=bool)
    positive_embeddings = emb_array[label_array]
    negative_embeddings = emb_array[~label_array]
    positive_centroid = positive_embeddings.mean(axis=0).astype(np.float32, copy=False)
    negative_centroid = negative_embeddings.mean(axis=0).astype(np.float32, copy=False)

    neutral_components = np.zeros((0, emb_array.shape[1]), dtype=np.float32)
    if use_neutral_pca:
        neutral_components = compute_neutral_components(
            emb_array,
            pc_count=neutral_pc_count,
            variance_target=neutral_pca_variance,
        )
        positive_centroid = project_out_components(positive_centroid, neutral_components)
        negative_centroid = project_out_components(negative_centroid, neutral_components)

    logic_vector = (positive_centroid - negative_centroid).astype(np.float32, copy=False)

    positive_path = artifact_root / "positive_centroid.npy"
    negative_path = artifact_root / "negative_centroid.npy"
    vector_path = artifact_root / "logic_vector.npy"
    neutral_path = artifact_root / "neutral_components.npy"
    metadata_path = artifact_root / "vector_metadata.json"

    np.save(positive_path, positive_centroid)
    np.save(negative_path, negative_centroid)
    np.save(vector_path, logic_vector)
    np.save(neutral_path, neutral_components)
    metadata_path.write_text(
        json.dumps(
            {
                "score_model_name": score_model_name,
                "layer_idx": int(layer_idx),
                "token_pool_start": int(token_pool_start),
                "example_count": int(len(texts)),
                "positive_count": int(label_array.sum()),
                "negative_count": int((~label_array).sum()),
                "text_fields": list(text_fields or ["thinking", "answer_content"]),
                "source_files": [str(Path(path)) for path in source_files],
                "neutral_component_count": int(neutral_components.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_dir": str(artifact_root),
        "positive_centroid": str(positive_path),
        "negative_centroid": str(negative_path),
        "logic_vector": str(vector_path),
        "neutral_components": str(neutral_path),
        "metadata": str(metadata_path),
    }


class LogicalVectorRewardProvider(IntrinsicRewardProvider):
    def __init__(
        self,
        *,
        artifact_dir: str,
        source_files: list[str] | None = None,
        score_model_name: str = "Qwen/Qwen3-0.6B",
        layer_idx: int = 8,
        token_pool_start: int = 8,
        batch_size: int = 4,
        text_fields: list[str] | None = None,
        label_field: str = "is_good",
        score_mode: str = "margin",
        reward_scale: float = 2.0,
        reward_clip: float | None = 1.0,
        empty_response_penalty: float = -1.0,
        device: str = "cpu",
        dtype: str | None = None,
        local_files_only: bool = False,
        max_length: int = 768,
        use_neutral_pca: bool = True,
        neutral_pca_variance: float = 0.5,
        neutral_pc_count: int | None = None,
    ) -> None:
        self.score_mode = score_mode
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.empty_response_penalty = float(empty_response_penalty)
        self.artifact_dir = Path(artifact_dir)

        metadata_path = self.artifact_dir / "vector_metadata.json"
        positive_path = self.artifact_dir / "positive_centroid.npy"
        negative_path = self.artifact_dir / "negative_centroid.npy"
        neutral_path = self.artifact_dir / "neutral_components.npy"

        if not (metadata_path.exists() and positive_path.exists() and negative_path.exists()):
            if not source_files:
                raise ValueError(
                    "Logical vector artifact files are missing and no source_files were provided to build them"
                )
            build_logical_vector_artifact(
                source_files=source_files,
                artifact_dir=self.artifact_dir,
                score_model_name=score_model_name,
                layer_idx=layer_idx,
                token_pool_start=token_pool_start,
                batch_size=batch_size,
                text_fields=text_fields,
                label_field=label_field,
                device=device,
                dtype=dtype,
                local_files_only=local_files_only,
                max_length=max_length,
                use_neutral_pca=use_neutral_pca,
                neutral_pca_variance=neutral_pca_variance,
                neutral_pc_count=neutral_pc_count,
            )

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.token_pool_start = int(self.metadata.get("token_pool_start", token_pool_start))
        self.positive_centroid = np.load(positive_path).astype(np.float32, copy=False)
        self.negative_centroid = np.load(negative_path).astype(np.float32, copy=False)
        self.neutral_components = (
            np.load(neutral_path).astype(np.float32, copy=False)
            if neutral_path.exists()
            else np.zeros((0, self.positive_centroid.shape[0]), dtype=np.float32)
        )
        self.scorer = _ResidualScorer(
            model_name=score_model_name,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            max_length=max_length,
        )
        if self.scorer.hidden_size != self.positive_centroid.shape[0]:
            raise ValueError(
                "Hidden-size mismatch between scorer model and logical vector centroid: "
                f"{self.scorer.hidden_size} != {self.positive_centroid.shape[0]}"
            )

    def _score_generation(self, generation: str) -> float:
        if not generation.strip():
            return self.empty_response_penalty
        hidden, attention = self.scorer.hidden_states([generation])
        embedding = _pool_embedding(
            hidden[0].detach().float().cpu().numpy(),
            attention[0].detach().cpu().numpy(),
            self.token_pool_start,
        )
        embedding = project_out_components(embedding, self.neutral_components)
        return _score_against_logic_centroids(
            embedding,
            self.positive_centroid,
            self.negative_centroid,
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
        del batch_records, metadata
        return [self._score_generation(generation) for generation in generations]
