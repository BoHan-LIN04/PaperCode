"""
简单的集成测试，演示 analyze-prompt 命令的使用。
"""

import json
from pathlib import Path
import tempfile
import torch

import pytest

from soft_prompt_repro.config import ExperimentConfig, load_experiment_config
from soft_prompt_repro.interpretability import (
    analyze_prompt_interpretability,
    compute_nearest_neighbors,
    extract_nearest_tokens_for_display,
)


def test_compute_nearest_neighbors():
    """测试最近邻计算功能。"""
    # 创建示例嵌入
    prompt_embeddings = torch.randn(5, 768)  # 5个 prompt tokens, 768维
    vocab_embeddings = torch.randn(30522, 768)  # T5 词汇表大小
    
    # 计算最近邻
    neighbors = compute_nearest_neighbors(prompt_embeddings, vocab_embeddings, k=5, metric="cosine")
    
    # 验证输出形状
    assert len(neighbors) == 5  # 5个 prompt tokens
    assert all(len(n) == 5 for n in neighbors)  # 每个都有5个最近邻
    
    # 验证 vocab_id 在合法范围内
    for token_neighbors in neighbors:
        for vocab_id, distance in token_neighbors:
            assert 0 <= vocab_id < 30522
            assert -2 <= distance <= 2  # cosine distance range (from -1 to 1, negated)


def test_nearest_neighbors_l2():
    """测试 L2 距离度量。"""
    prompt_embeddings = torch.randn(3, 256)
    vocab_embeddings = torch.randn(1000, 256)
    
    neighbors = compute_nearest_neighbors(prompt_embeddings, vocab_embeddings, k=3, metric="l2")
    
    assert len(neighbors) == 3
    assert all(len(n) == 3 for n in neighbors)


def test_interpretability_analysis_output_format():
    """测试可解释性分析的输出格式。"""
    # 创建示例数据
    sample_neighbors = [
        [
            {"token": "task", "vocab_id": 100, "distance": 0.1},
            {"token": "problem", "vocab_id": 101, "distance": 0.2},
            {"token": "objective", "vocab_id": 102, "distance": 0.3},
            {"token": "goal", "vocab_id": 103, "distance": 0.4},
            {"token": "target", "vocab_id": 104, "distance": 0.5},
        ],
        [
            {"token": "answer", "vocab_id": 200, "distance": 0.15},
            {"token": "response", "vocab_id": 201, "distance": 0.25},
            {"token": "output", "vocab_id": 202, "distance": 0.35},
            {"token": "result", "vocab_id": 203, "distance": 0.45},
            {"token": "solution", "vocab_id": 204, "distance": 0.55},
        ],
    ]
    
    # 创建示例分析结果
    analysis_result = {
        "neighbors": sample_neighbors,
        "analysis": {
            "semantic_clustering": "观察到紧密的语义簇",
            "class_label_persistence": "找到类标签",
            "duplicate_neighbors": "无重复",
            "summary": "提示学习了词汇语义表示"
        },
        "prompt_size": 20,
        "init_strategy": "class_labels",
    }
    
    # 测试格式化显示
    display = extract_nearest_tokens_for_display(analysis_result, max_tokens=2)
    
    assert "Prompt Interpretability Analysis" in display
    assert "Prompt size: 20" in display
    assert "Init strategy: class_labels" in display
    assert "Token 0:" in display
    assert "task" in display
    assert "answer" in display


if __name__ == "__main__":
    test_compute_nearest_neighbors()
    test_nearest_neighbors_l2()
    test_interpretability_analysis_output_format()
    print("✅ 所有可解释性测试通过！")
