from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from anthropic_emotions_repro.config import load_config, resolve_artifact_root
from anthropic_emotions_repro.data.activation_cache import ActivationCacheReader
from anthropic_emotions_repro.io import read_json, write_json
from anthropic_emotions_repro.model.qwen import QwenHookedModel
from anthropic_emotions_repro.runtime import decode_token_piece, format_chat_prompt, is_readable_english_token, last_valid_index
from anthropic_emotions_repro.steering.intervention import ResidualVectorIntervener
from anthropic_emotions_repro.utils import set_seed


PREFERRED_ANALYSIS_EMOTIONS = [
    "desperate",
    "sad",
    "grief",
    "fearful",
    "anxious",
    "joyful",
    "excited",
    "happy",
    "enthusiastic",
    "calm",
    "hopeful",
    "grateful",
]


def _ensure_experiment_dir(root: Path, name: str) -> Path:
    exp = root / name
    (exp / "tables").mkdir(parents=True, exist_ok=True)
    (exp / "figures").mkdir(parents=True, exist_ok=True)
    return exp


def _select_emotions(emotion_names: list[str], k: int = 6) -> list[str]:
    selected = [name for name in PREFERRED_ANALYSIS_EMOTIONS if name in emotion_names]
    if len(selected) < k:
        for name in emotion_names:
            if name not in selected:
                selected.append(name)
            if len(selected) >= k:
                break
    return selected[:k]


def _load_vectors(artifact_root: Path) -> tuple[np.ndarray, list[str]]:
    vectors = np.load(artifact_root / "05_emotion_vectors" / "intermediate" / "emotion_vectors_orth.npy")
    meta = read_json(artifact_root / "05_emotion_vectors" / "intermediate" / "vector_metadata.json")
    names = meta["emotion_names"]
    return vectors.astype(np.float32), names


def _sample_embeddings(cache: ActivationCacheReader, sample_index: pd.DataFrame) -> np.ndarray:
    acts = np.asarray(cache.activations, dtype=np.float32)
    sample_ids = np.asarray(cache.sample_ids)
    positions = np.asarray(cache.token_positions)
    out = []
    for row in sample_index.itertuples(index=False):
        mask = (sample_ids == row.sample_id) & (positions >= row.pool_start_token)
        if not np.any(mask):
            mask = sample_ids == row.sample_id
        out.append(acts[mask].mean(axis=0))
    return np.stack(out, axis=0)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _local_activation_analysis(
    artifact_root: Path,
    analysis_root: Path,
    cfg,
    emotion_names: list[str],
    vectors: np.ndarray,
) -> dict[str, str]:
    exp_root = _ensure_experiment_dir(analysis_root, "01_local_activation")
    cache = ActivationCacheReader(artifact_root / "04_activation_cache" / "raw")
    sample_index = pd.read_parquet(artifact_root / "04_activation_cache" / "tables" / "sample_index.parquet")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, local_files_only=True)
    sample_embeds = _sample_embeddings(cache, sample_index)
    name_to_idx = {name: idx for idx, name in enumerate(emotion_names)}
    acts = np.asarray(cache.activations, dtype=np.float32)
    sample_ids = np.asarray(cache.sample_ids)
    token_ids = np.asarray(cache.token_ids)
    token_pos = np.asarray(cache.token_positions)

    selected = _select_emotions(emotion_names, k=min(6, len(emotion_names)))
    summary_rows = []
    md_lines = ["# Experiment 1: Local Activation on Training Stories", ""]

    for emotion in selected:
        vec = vectors[name_to_idx[emotion]]
        subset = sample_index[sample_index["emotion"] == emotion].copy()
        idxs = subset.index.to_numpy()
        scores = sample_embeds[idxs] @ vec
        best_local = int(np.argmax(scores))
        row = subset.iloc[best_local]
        mask = sample_ids == int(row["sample_id"])
        tok_scores = acts[mask] @ vec
        toks = [decode_token_piece(tokenizer, x) for x in token_ids[mask]]
        positions = token_pos[mask]
        q = float(np.quantile(tok_scores, 0.90)) if len(tok_scores) > 1 else float(tok_scores[0])
        top_mask = tok_scores >= q
        abs_scores = np.abs(tok_scores)
        k_top = max(1, math.ceil(len(abs_scores) * 0.1))
        top_share = float(np.sort(abs_scores)[-k_top:].sum() / max(abs_scores.sum(), 1e-8))
        df = pd.DataFrame(
            {
                "token_position": positions,
                "token": toks,
                "score": tok_scores,
                "highlight": top_mask.astype(int),
            }
        )
        token_csv = exp_root / "tables" / f"tokens_{emotion}.csv"
        df.to_csv(token_csv, index=False)

        fig_path = exp_root / "figures" / f"activation_{emotion}.png"
        plt.figure(figsize=(10, 3))
        colors = ["crimson" if hi else "steelblue" for hi in top_mask]
        plt.bar(positions, tok_scores, color=colors)
        plt.axhline(q, color="black", linestyle="--", linewidth=1, label="90th pct threshold")
        plt.title(f"Local token activation: {emotion}")
        plt.xlabel("token position")
        plt.ylabel("projection score")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

        highlighted = []
        for token, hi in zip(toks, top_mask):
            highlighted.append(f"[[{token}]]" if hi else token)
        snippet = " ".join(highlighted[:80])
        summary_rows.append(
            {
                "emotion": emotion,
                "sample_id": int(row["sample_id"]),
                "topic_id": int(row["topic_id"]),
                "token_count": int(len(tok_scores)),
                "max_score": float(tok_scores.max()),
                "top10pct_abs_share": top_share,
                "highlight_threshold": q,
                "token_csv": str(token_csv),
                "figure": str(fig_path),
            }
        )
        md_lines.extend(
            [
                f"## {emotion}",
                "",
                f"- sample_id: `{int(row['sample_id'])}`",
                f"- topic_id: `{int(row['topic_id'])}`",
                f"- top 10% absolute activation share: `{top_share:.3f}`",
                f"- max token score: `{float(tok_scores.max()):.4f}`",
                "- 解释：如果这个值集中在少量 token 上，说明向量更像是在局部表达/推断情绪位置激活，而不是全文均匀抬高。",
                "",
                snippet,
                "",
            ]
        )

    summary_path = exp_root / "tables" / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    report_path = exp_root / "README.md"
    _write_text(report_path, "\n".join(md_lines) + "\n")
    return {
        "local_activation_summary": str(summary_path),
        "local_activation_report": str(report_path),
    }


def _single_token_candidates(tokenizer, emotion_names: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for name in emotion_names:
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) == 1:
            out.append((name, ids[0]))
    return out


def _next_token_logits(model, encoded) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(**encoded).logits
    idx = last_valid_index(encoded["attention_mask"][0])
    return logits[0, idx, :]


def _steering_analysis(
    artifact_root: Path,
    analysis_root: Path,
    cfg,
    emotion_names: list[str],
    vectors: np.ndarray,
) -> dict[str, str]:
    exp_root = _ensure_experiment_dir(analysis_root, "02_steering_validation")
    qwen = QwenHookedModel(
        model_name=cfg.model_name,
        layer_idx=cfg.main_layer,
        dtype=cfg.runtime.dtype,
        attn_impl=cfg.runtime.attn_impl,
        device_map="auto",
        use_cache=False,
    )
    tokenizer = qwen.tokenizer
    prompt = format_chat_prompt(tokenizer, "How does he feel? Answer with exactly one emotion word.", enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(qwen.model.device)
    baseline_logits = _next_token_logits(qwen.model, encoded)
    candidates = _single_token_candidates(tokenizer, emotion_names)
    selected = [name for name, _ in candidates if name in _select_emotions(emotion_names, k=min(6, len(emotion_names)))]
    candidate_map = {name: token_id for name, token_id in candidates if name in selected}
    baseline_probs = torch.softmax(baseline_logits, dim=-1)

    rows = []
    strength = 2.0
    for emotion in selected:
        vec = torch.tensor(vectors[emotion_names.index(emotion)])
        intervener = ResidualVectorIntervener(qwen.hooked_layer, vec, strength=strength)
        with intervener.apply():
            steered_logits = _next_token_logits(qwen.model, encoded)
        steered_probs = torch.softmax(steered_logits, dim=-1)
        for candidate, token_id in candidate_map.items():
            rows.append(
                {
                    "steer_emotion": emotion,
                    "candidate_emotion": candidate,
                    "baseline_prob": float(baseline_probs[token_id].item()),
                    "steered_prob": float(steered_probs[token_id].item()),
                    "delta_prob": float((steered_probs[token_id] - baseline_probs[token_id]).item()),
                }
            )

    df = pd.DataFrame(rows)
    out_path = exp_root / "tables" / "steering_validation.csv"
    df.to_csv(out_path, index=False)

    pivot = df.pivot(index="steer_emotion", columns="candidate_emotion", values="delta_prob").fillna(0.0)
    fig_heat = exp_root / "figures" / "delta_prob_heatmap.png"
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.to_numpy(), cmap="coolwarm", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="delta probability")
    plt.title("Steering effect on candidate emotion probabilities")
    plt.tight_layout()
    plt.savefig(fig_heat, dpi=160)
    plt.close()

    fig_bar = exp_root / "figures" / "target_delta_bar.png"
    target_df = df[df["steer_emotion"] == df["candidate_emotion"]]
    plt.figure(figsize=(8, 4))
    plt.bar(target_df["steer_emotion"], target_df["delta_prob"], color="darkgreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Delta probability")
    plt.title("Target emotion probability under steering")
    plt.tight_layout()
    plt.savefig(fig_bar, dpi=160)
    plt.close()

    mean_target_delta = float(target_df["delta_prob"].mean()) if not target_df.empty else 0.0
    report_path = exp_root / "README.md"
    _write_text(
        report_path,
        "\n".join(
            [
                "# Experiment 2: Simple Steering Validation",
                "",
                f"- prompt: `How does he feel? Answer with exactly one emotion word.`",
                f"- steering strength: `{strength}` residual units",
                f"- analyzed emotions: `{', '.join(selected)}`",
                f"- mean target delta probability: `{mean_target_delta:.6e}`",
                "- 解释：如果 steer 到某个情绪后，同名候选词的 next-token 概率上升，而不匹配候选下降或相对更弱，就说明向量具备简单因果效应。",
                "",
            ]
        )
        + "\n",
    )
    return {
        "steering_validation": str(out_path),
        "steering_report": str(report_path),
    }


def _logit_lens_analysis(
    artifact_root: Path,
    analysis_root: Path,
    cfg,
    emotion_names: list[str],
    vectors: np.ndarray,
) -> dict[str, str]:
    exp_root = _ensure_experiment_dir(analysis_root, "03_logit_lens")
    qwen = QwenHookedModel(
        model_name=cfg.model_name,
        layer_idx=cfg.main_layer,
        dtype=cfg.runtime.dtype,
        attn_impl=cfg.runtime.attn_impl,
        device_map="auto",
        use_cache=False,
    )
    tokenizer = qwen.tokenizer
    weight = qwen.model.lm_head.weight.detach().float().cpu().numpy()
    selected = _select_emotions(emotion_names, k=min(6, len(emotion_names)))
    rows = []
    md_lines = ["# Experiment 3: Logit Lens / Unembed Validation", ""]
    for emotion in selected:
        vec = vectors[emotion_names.index(emotion)]
        scores = weight @ vec
        filtered = []
        order = np.argsort(scores)[::-1]
        for token_id in order:
            token = decode_token_piece(tokenizer, int(token_id))
            if is_readable_english_token(token):
                filtered.append((int(token_id), token, float(scores[token_id])))
            if len(filtered) >= 20:
                break
        emotion_rows = []
        for rank, (token_id, token, score) in enumerate(filtered, start=1):
            row = {
                "emotion": emotion,
                "rank": rank,
                "token_id": token_id,
                "token": token,
                "score": score,
            }
            rows.append(row)
            emotion_rows.append(row)

        fig_path = exp_root / "figures" / f"logit_lens_{emotion}.png"
        top10 = emotion_rows[:10]
        plt.figure(figsize=(9, 4))
        plt.bar([r["token"] for r in top10], [r["score"] for r in top10], color="purple")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("logit-lens score")
        plt.title(f"Top upweighted tokens: {emotion}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

        md_lines.extend(
            [
                f"## {emotion}",
                "",
                "- Top tokens:",
                ", ".join(f"`{r['token']}`" for r in top10),
                "",
            ]
        )

    out_path = exp_root / "tables" / "logit_lens_top_tokens.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    report_path = exp_root / "README.md"
    _write_text(report_path, "\n".join(md_lines) + "\n")
    return {
        "logit_lens_top_tokens": str(out_path),
        "logit_lens_report": str(report_path),
    }


def _cosine_analysis(
    artifact_root: Path,
    analysis_root: Path,
    emotion_names: list[str],
    vectors: np.ndarray,
) -> dict[str, str]:
    exp_root = _ensure_experiment_dir(analysis_root, "04_cosine_structure")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    cos = (vectors @ vectors.T) / np.clip(norms @ norms.T, 1e-8, None)
    matrix_path = exp_root / "tables" / "emotion_cosine_similarity.csv"
    pd.DataFrame(cos, index=emotion_names, columns=emotion_names).to_csv(matrix_path)

    nn_rows = []
    for i, emotion in enumerate(emotion_names):
        order = np.argsort(cos[i])[::-1]
        neighbors = [j for j in order if j != i][:5]
        row = {"emotion": emotion}
        for rank, j in enumerate(neighbors, start=1):
            row[f"nn{rank}"] = emotion_names[j]
            row[f"sim{rank}"] = float(cos[i, j])
        nn_rows.append(row)
    nn_path = exp_root / "tables" / "emotion_nearest_neighbors.csv"
    pd.DataFrame(nn_rows).to_csv(nn_path, index=False)

    fig_path = exp_root / "figures" / "emotion_cosine_heatmap.png"
    plt.figure(figsize=(10, 8))
    plt.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(emotion_names)), emotion_names, rotation=90, fontsize=7)
    plt.yticks(range(len(emotion_names)), emotion_names, fontsize=7)
    plt.colorbar(label="cosine similarity")
    plt.title("Emotion vector cosine similarity")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    report_path = exp_root / "README.md"
    preview = pd.DataFrame(nn_rows).head(8)
    preview_lines = []
    for row in preview.to_dict(orient="records"):
        preview_lines.append(
            f"- {row['emotion']}: "
            f"{row.get('nn1', '')} ({row.get('sim1', 0.0):.3f}), "
            f"{row.get('nn2', '')} ({row.get('sim2', 0.0):.3f}), "
            f"{row.get('nn3', '')} ({row.get('sim3', 0.0):.3f})"
        )
    _write_text(
        report_path,
        "# Experiment 4: Cosine Structure\n\n"
        "这个实验计算所有情绪向量的两两余弦相似度，并导出最近邻关系。"
        "如果结构合理，语义相近的情绪应该互为最近邻，不同簇之间相似度会较低。\n\n"
        + "\n".join(preview_lines)
        + "\n",
    )
    return {
        "emotion_cosine_similarity": str(matrix_path),
        "emotion_nearest_neighbors": str(nn_path),
        "cosine_report": str(report_path),
    }


def _root_report(
    analysis_root: Path,
    cfg,
    artifact_root: Path,
    emotion_names: list[str],
    local_outputs: dict[str, str],
    steering_outputs: dict[str, str],
    logit_outputs: dict[str, str],
    cosine_outputs: dict[str, str],
) -> Path:
    selected = _select_emotions(emotion_names, k=min(6, len(emotion_names)))
    has_negative = any(name in emotion_names for name in ["desperate", "sad", "fearful", "anxious", "grief"])
    lines = [
        "# Analysis",
        "",
        "## Scope",
        "",
        f"- artifact: `{artifact_root}`",
        f"- model: `{cfg.model_name}`",
        f"- layer: `{cfg.main_layer}`",
        f"- emotion_count: `{len(emotion_names)}`",
        f"- analyzed_emotions: `{', '.join(selected)}`",
        "",
        "## Folder Layout",
        "",
        "- `01_local_activation/`",
        "- `02_steering_validation/`",
        "- `03_logit_lens/`",
        "- `04_cosine_structure/`",
        "",
        "## Notes",
        "",
        "- 当前 real_small 向量集主要由正向情绪组成，不包含 desperate/sad/fear/anxiety/grief 等负向情绪，因此经典负向案例不能在当前 artifact 中直接展示。",
        f"- negative_emotions_available: `{has_negative}`",
        "",
        "## Key Outputs",
        "",
        f"- local activation summary: `{local_outputs['local_activation_summary']}`",
        f"- steering validation: `{steering_outputs['steering_validation']}`",
        f"- logit lens top tokens: `{logit_outputs['logit_lens_top_tokens']}`",
        f"- cosine neighbors: `{cosine_outputs['emotion_nearest_neighbors']}`",
        "",
    ]
    report_path = analysis_root / "README.md"
    _write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vector analysis experiments and write outputs into artifact_root/Analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.artifact_root is not None:
        cfg.artifact_root = args.artifact_root
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)
    artifact_root = resolve_artifact_root(cfg)
    analysis_root = artifact_root / "Analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_json(analysis_root / "config.resolved.json", cfg.to_dict())

    vectors, emotion_names = _load_vectors(artifact_root)
    local_outputs = _local_activation_analysis(artifact_root, analysis_root, cfg, emotion_names, vectors)
    steering_outputs = _steering_analysis(artifact_root, analysis_root, cfg, emotion_names, vectors)
    logit_outputs = _logit_lens_analysis(artifact_root, analysis_root, cfg, emotion_names, vectors)
    cosine_outputs = _cosine_analysis(artifact_root, analysis_root, emotion_names, vectors)
    report_path = _root_report(analysis_root, cfg, artifact_root, emotion_names, local_outputs, steering_outputs, logit_outputs, cosine_outputs)

    write_json(
        analysis_root / "manifest.json",
        {
            "status": "completed",
            "command": "run_vector_analysis",
            "artifact_root": str(artifact_root),
            "outputs": {
                **local_outputs,
                **steering_outputs,
                **logit_outputs,
                **cosine_outputs,
                "report": str(report_path),
            },
        },
    )
    write_json(
        analysis_root / "metrics.json",
        {
            "emotion_count": len(emotion_names),
            "analyzed_emotion_count": len(_select_emotions(emotion_names)),
        },
    )


if __name__ == "__main__":
    main()
