import torch
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer

# 1. 下游任务指标（如准确率、F1、BLEU、ROUGE）
def compute_accuracy(preds: List[str], labels: List[str]) -> float:
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(preds) if preds else 0.0

def compute_bleu(preds: List[str], labels: List[str]) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu
        return corpus_bleu([[l.split()] for l in labels], [p.split() for p in preds])
    except ImportError:
        return -1.0

# 2. prompt token 最近邻分析
def prompt_token_nearest_neighbors(
    soft_prompt: torch.Tensor,
    embedding_matrix: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 5
) -> Dict[int, List[str]]:
    """
    对每个 prompt token，找最近邻的真实 token。
    soft_prompt: [num_virtual_tokens, hidden_size]
    embedding_matrix: [vocab_size, hidden_size]
    """
    soft_prompt = soft_prompt.detach().cpu().numpy()
    embedding_matrix = embedding_matrix.detach().cpu().numpy()
    neighbors = {}
    for i, vec in enumerate(soft_prompt):
        sims = embedding_matrix @ vec / (np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(vec) + 1e-8)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        tokens = [tokenizer.decode([idx]) for idx in top_idx]
        neighbors[i] = tokens
    return neighbors

# 3. sweep/ablation 自动可视化（复用 plotting.py）
# 见 plotting.py 的 plot_figure_from_csv

# 用法示例：
# acc = compute_accuracy(preds, labels)
# bleu = compute_bleu(preds, labels)
# neighbors = prompt_token_nearest_neighbors(soft_prompt, model.get_input_embeddings().weight, tokenizer)
# print(neighbors)
