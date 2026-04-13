from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from anthropic_emotions_repro.data.activation_cache import ActivationCacheWriter
from anthropic_emotions_repro.io import read_jsonl, write_json
from anthropic_emotions_repro.model.qwen import QwenHookedModel
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs
from anthropic_emotions_repro.pipeline.generate_emotion_corpus import expected_corpus_counts, promote_legacy_corpus_inplace
from anthropic_emotions_repro.pipeline.stub_repr import stub_token_activations
from anthropic_emotions_repro.runtime import positions_for_tensor, valid_token_positions


def _count_tokens_stub(texts: list[str]) -> int:
    return sum(max(1, len(text.strip().split())) for text in texts)


def _count_tokens_real(tokenizer, texts: list[str], max_length: int) -> int:
    total = 0
    for text in texts:
        encoded = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        total += len(encoded["input_ids"])
    return total


def _stub_run(cfg, workspace: Path, synthetic_root: Path) -> dict[str, str]:
    stories = read_jsonl(synthetic_root / "raw" / "stories_train.jsonl")
    neutral_stories = read_jsonl(synthetic_root / "raw" / "neutral_stories.jsonl")
    hidden_size = 512
    total_tokens = _count_tokens_stub([row["text"] for row in stories])
    cache = ActivationCacheWriter(workspace / "raw", num_tokens=total_tokens, hidden_size=hidden_size)
    sample_rows = []
    offset = 0
    for row in tqdm(stories, desc="extract_residuals:stub_stories", total=len(stories), dynamic_ncols=True):
        activations, tokens = stub_token_activations(row["text"], hidden_size=hidden_size)
        token_ids = np.arange(len(tokens), dtype=np.int32)
        token_positions = np.arange(len(tokens), dtype=np.int32)
        sample_ids = np.full((len(tokens),), int(row["sample_id"]), dtype=np.int32)
        cache.write_batch(activations, sample_ids, token_ids, token_positions)
        sample_rows.append(
            {
                "sample_id": row["sample_id"],
                "emotion": row["emotion"],
                "topic_id": row["topic_id"],
                "text": row["text"],
                "token_count": len(tokens),
                "pool_start_token": cfg.vector_extraction.token_pool_start,
                "cache_offset_start": offset,
                "cache_offset_end": offset + len(tokens),
            }
        )
        offset += len(tokens)
    cache.flush()
    cache.write_metadata({"layer_idx": cfg.main_layer, "model_name": cfg.model_name})
    neutral_embeds = [
        stub_token_activations(row["text"], hidden_size=hidden_size)[0].mean(axis=0)
        for row in tqdm(neutral_stories, desc="extract_residuals:stub_neutral", total=len(neutral_stories), dynamic_ncols=True)
    ]
    neutral_array = np.stack(neutral_embeds, axis=0)
    np.save(workspace / "intermediate" / "neutral_story_embeddings.npy", neutral_array)
    sample_index_path = workspace / "tables" / "sample_index.parquet"
    pd.DataFrame(sample_rows).to_parquet(sample_index_path, index=False)
    tokens_path = workspace / "tables" / "tokens.parquet"
    pd.DataFrame(
        {
            "sample_id": np.asarray(cache._sample_ids[: cache.offset]),
            "token_id": np.asarray(cache._token_ids[: cache.offset]),
            "token_position": np.asarray(cache._positions[: cache.offset]),
        }
    ).to_parquet(tokens_path, index=False)
    metadata_path = workspace / "raw" / "metadata.json"
    write_json(metadata_path, {"hidden_size": hidden_size, "written_tokens": cache.offset, "layer_idx": cfg.main_layer})
    return {
        "activation_cache_root": str(workspace / "raw"),
        "sample_index": str(sample_index_path),
        "tokens": str(tokens_path),
        "metadata": str(metadata_path),
        "neutral_embeddings": str(workspace / "intermediate" / "neutral_story_embeddings.npy"),
    }


def run(cfg, workspace: Path, synthetic_root: Path) -> dict[str, str]:
    if cfg.use_stub_data:
        return _stub_run(cfg, workspace, synthetic_root)

    expected_stories, expected_neutral = expected_corpus_counts(cfg)
    current_stories = read_jsonl(synthetic_root / "raw" / "stories_train.jsonl")
    current_neutral = read_jsonl(synthetic_root / "raw" / "neutral_stories.jsonl")
    current_rejected_stories = read_jsonl(synthetic_root / "raw" / "rejected_stories.jsonl") if (synthetic_root / "raw" / "rejected_stories.jsonl").exists() else []
    current_rejected_neutral = read_jsonl(synthetic_root / "raw" / "rejected_neutral_stories.jsonl") if (synthetic_root / "raw" / "rejected_neutral_stories.jsonl").exists() else []
    if (
        (len(current_stories) < expected_stories or len(current_neutral) < expected_neutral)
        and (current_rejected_stories or current_rejected_neutral)
    ):
        promote_legacy_corpus_inplace(
            cfg,
            synthetic_root,
            synthetic_root.parent / "01_topic_bank",
            synthetic_root.parent / "02_prompt_templates",
        )

    stories = read_jsonl(synthetic_root / "raw" / "stories_train.jsonl")
    neutral_stories = read_jsonl(synthetic_root / "raw" / "neutral_stories.jsonl")
    if not neutral_stories:
        raise RuntimeError(
            "neutral_stories.jsonl is empty, so neutral PCA cannot be estimated. "
            "Please inspect 03_synthetic_emotion_corpus/tables/corpus_quality_summary.csv "
            "and the recovered/fallback story files in 03_synthetic_emotion_corpus/raw."
        )

    qwen = QwenHookedModel(
        model_name=cfg.model_name,
        layer_idx=cfg.main_layer,
        dtype=cfg.runtime.dtype,
        attn_impl=cfg.runtime.attn_impl,
        device_map="auto",
        use_cache=False,
    )
    hidden_size = qwen.hidden_size
    total_tokens = _count_tokens_real(qwen.tokenizer, [row["text"] for row in stories], max_length=cfg.runtime.max_length)
    cache = ActivationCacheWriter(workspace / "raw", num_tokens=total_tokens, hidden_size=hidden_size)

    sample_rows = []
    offset = 0
    texts = [row["text"] for row in stories]
    for start in tqdm(
        range(0, len(stories), cfg.runtime.extraction_batch_size),
        total=(len(stories) + cfg.runtime.extraction_batch_size - 1) // cfg.runtime.extraction_batch_size,
        desc="extract_residuals:stories",
        dynamic_ncols=True,
    ):
        batch_rows = stories[start : start + cfg.runtime.extraction_batch_size]
        batch_texts = texts[start : start + cfg.runtime.extraction_batch_size]
        inputs = qwen.encode_batch(batch_texts, max_length=cfg.runtime.max_length)
        hidden = qwen.forward(inputs)
        attention = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"]
        for idx, row in enumerate(batch_rows):
            positions = valid_token_positions(attention[idx].detach().cpu())
            valid = int(positions.numel())
            activations = (
                hidden[idx]
                .index_select(0, positions_for_tensor(positions, hidden[idx]))
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            token_ids = (
                input_ids[idx]
                .index_select(0, positions_for_tensor(positions, input_ids[idx]))
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32, copy=False)
            )
            token_positions = np.arange(valid, dtype=np.int32)
            sample_ids = np.full((valid,), int(row["sample_id"]), dtype=np.int32)
            cache.write_batch(activations, sample_ids, token_ids, token_positions)
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "emotion": row["emotion"],
                    "topic_id": row["topic_id"],
                    "text": row["text"],
                    "token_count": valid,
                    "pool_start_token": cfg.vector_extraction.token_pool_start,
                    "cache_offset_start": offset,
                    "cache_offset_end": offset + valid,
                }
            )
            offset += valid

    cache.flush()
    cache.write_metadata({"layer_idx": cfg.main_layer, "model_name": cfg.model_name})

    neutral_embeds = []
    neutral_texts = [row["text"] for row in neutral_stories]
    for start in tqdm(
        range(0, len(neutral_texts), cfg.runtime.extraction_batch_size),
        total=(len(neutral_texts) + cfg.runtime.extraction_batch_size - 1) // cfg.runtime.extraction_batch_size,
        desc="extract_residuals:neutral",
        dynamic_ncols=True,
    ):
        batch = neutral_texts[start : start + cfg.runtime.extraction_batch_size]
        inputs = qwen.encode_batch(batch, max_length=cfg.runtime.max_length)
        hidden = qwen.forward(inputs)
        attention = inputs["attention_mask"].bool()
        for idx in range(hidden.shape[0]):
            positions = valid_token_positions(attention[idx].detach().cpu())
            neutral_embeds.append(
                hidden[idx]
                .index_select(0, positions_for_tensor(positions, hidden[idx]))
                .detach()
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )
    neutral_array = np.stack(neutral_embeds, axis=0)
    np.save(workspace / "intermediate" / "neutral_story_embeddings.npy", neutral_array)

    sample_index_path = workspace / "tables" / "sample_index.parquet"
    pd.DataFrame(sample_rows).to_parquet(sample_index_path, index=False)
    tokens_path = workspace / "tables" / "tokens.parquet"
    pd.DataFrame(
        {
            "sample_id": np.asarray(cache._sample_ids[: cache.offset]),
            "token_id": np.asarray(cache._token_ids[: cache.offset]),
            "token_position": np.asarray(cache._positions[: cache.offset]),
        }
    ).to_parquet(tokens_path, index=False)
    metadata_path = workspace / "raw" / "metadata.json"
    write_json(metadata_path, {"hidden_size": hidden_size, "written_tokens": cache.offset, "layer_idx": cfg.main_layer, "model_name": cfg.model_name})
    return {
        "activation_cache_root": str(workspace / "raw"),
        "sample_index": str(sample_index_path),
        "tokens": str(tokens_path),
        "metadata": str(metadata_path),
        "neutral_embeddings": str(workspace / "intermediate" / "neutral_story_embeddings.npy"),
    }


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Extract residual activations from synthetic stories and neutral stories")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("extract_residuals", args)
    outputs = run(cfg, workspace.root, artifact_root / "03_synthetic_emotion_corpus")
    save_step_outputs(
        workspace,
        command_name="extract_residuals",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步只读取生成好的情绪故事与中性故事，不再依赖任何 held-out 文档或外部 neutral 语料。",
        output_summary=f"主 residual cache 写入 `{workspace.raw}`，sample index 与 token 表写入 `{workspace.tables}`，中性故事 embedding 写入 `{workspace.intermediate}`。",
        technique_summary="默认路径真实加载 Qwen3，在指定层抽取 residual stream；中性故事平均向量用于后续 PCA 去噪。",
        metrics={"main_layer": cfg.main_layer, "token_pool_start": cfg.vector_extraction.token_pool_start},
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
