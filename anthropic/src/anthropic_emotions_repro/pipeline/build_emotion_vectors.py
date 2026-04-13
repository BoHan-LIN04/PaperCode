from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from anthropic_emotions_repro.data.activation_cache import ActivationCacheReader
from anthropic_emotions_repro.io import write_json
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs


def _sample_means(cache: ActivationCacheReader, sample_index: pd.DataFrame) -> np.ndarray:
    rows = []
    acts = np.asarray(cache.activations, dtype=np.float32)
    sample_ids = np.asarray(cache.sample_ids)
    positions = np.asarray(cache.token_positions)
    for row in tqdm(
        sample_index.itertuples(index=False),
        desc="build_emotion_vectors:sample_pooling",
        total=len(sample_index),
        dynamic_ncols=True,
    ):
        mask = (sample_ids == row.sample_id) & (positions >= row.pool_start_token)
        if not np.any(mask):
            mask = sample_ids == row.sample_id
        rows.append(acts[mask].mean(axis=0))
    return np.stack(rows, axis=0)


def _orthogonalize(vectors: np.ndarray, neutral_embeddings: np.ndarray, variance_target: float) -> tuple[np.ndarray, int]:
    pca_full = PCA(random_state=42)
    pca_full.fit(neutral_embeddings)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, variance_target) + 1)
    components = pca_full.components_[:k]
    projected = vectors @ components.T @ components
    return vectors - projected, k


def run(cfg, workspace: Path, cache_root: Path) -> dict[str, str]:
    cache = ActivationCacheReader(cache_root / "raw")
    sample_index = pd.read_parquet(cache_root / "tables" / "sample_index.parquet")
    neutral_embeddings = np.load(cache_root / "intermediate" / "neutral_story_embeddings.npy")

    sample_means = _sample_means(cache, sample_index)
    grouped = sample_index.assign(_idx=np.arange(len(sample_index))).groupby("emotion")["_idx"].apply(list)
    emotion_names = sorted(grouped.index.tolist())
    raw_vectors = np.stack(
        [
            sample_means[grouped[emotion]].mean(axis=0)
            for emotion in tqdm(emotion_names, desc="build_emotion_vectors:emotion_averaging", total=len(emotion_names), dynamic_ncols=True)
        ],
        axis=0,
    )
    raw_vectors = raw_vectors - raw_vectors.mean(axis=0, keepdims=True)
    orth_vectors, pc_count = _orthogonalize(raw_vectors, neutral_embeddings, cfg.vector_extraction.neutral_pca_variance)

    raw_path = workspace / "intermediate" / "emotion_vectors_raw.npy"
    orth_path = workspace / "intermediate" / "emotion_vectors_orth.npy"
    stats_path = workspace / "tables" / "emotion_stats.csv"
    meta_path = workspace / "intermediate" / "vector_metadata.json"

    np.save(raw_path, raw_vectors)
    np.save(orth_path, orth_vectors)
    pd.DataFrame(
        {
            "emotion": emotion_names,
            "raw_norm": np.linalg.norm(raw_vectors, axis=1),
            "orth_norm": np.linalg.norm(orth_vectors, axis=1),
        }
    ).to_csv(stats_path, index=False)
    write_json(
        meta_path,
        {
            "emotion_names": emotion_names,
            "pc_count_removed": pc_count,
            "variance_target": cfg.vector_extraction.neutral_pca_variance,
            "token_pool_start": cfg.vector_extraction.token_pool_start,
        },
    )

    return {
        "emotion_vectors_raw": str(raw_path),
        "emotion_vectors_orth": str(orth_path),
        "emotion_stats": str(stats_path),
        "vector_metadata": str(meta_path),
    }


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Build emotion vectors from synthetic stories and neutral stories")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("build_emotion_vectors", args)
    outputs = run(cfg, workspace.root, artifact_root / "04_activation_cache")
    save_step_outputs(
        workspace,
        command_name="build_emotion_vectors",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步读取故事 residual cache 与 neutral story embeddings，按样本平均、按情绪平均后构建向量。",
        output_summary=f"raw/orth 两套 emotion vectors 写入 `{workspace.intermediate}`，统计写入 `{workspace.tables}`。",
        technique_summary="向量定义固定为：token>=start 池化 -> emotion 平均 -> 跨情绪中心化 -> neutral PCA 去噪。",
        metrics={"emotion_count": cfg.story_generation.emotion_count, "neutral_pca_variance": cfg.vector_extraction.neutral_pca_variance},
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
