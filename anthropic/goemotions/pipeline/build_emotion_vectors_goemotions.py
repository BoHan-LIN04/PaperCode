from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json

def _orthogonalize(vectors: np.ndarray, neutral_embeddings: np.ndarray, variance_target: float):
    pca_full = PCA(random_state=42)
    pca_full.fit(neutral_embeddings)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, variance_target) + 1)
    components = pca_full.components_[:k]
    projected = vectors @ components.T @ components
    return vectors - projected, k

# 配置
CACHE_DIR = "../artifacts/activation_cache"
OUTPUT_DIR = "../artifacts/emotion_vectors"
VARIANCE_TARGET = 0.5

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 读取 activation cache
acts = np.load(f"{CACHE_DIR}/activations.npy")
sample_ids = np.load(f"{CACHE_DIR}/sample_ids.npy")
token_positions = np.load(f"{CACHE_DIR}/token_positions.npy")
sample_index = pd.read_parquet(f"{CACHE_DIR}/sample_index.parquet")

# 按样本池化
sample_means = []
for i, row in sample_index.iterrows():
    mask = (sample_ids == row.sample_id) & (token_positions >= row.pool_start_token)
    if not np.any(mask):
        mask = sample_ids == row.sample_id
    sample_means.append(acts[mask].mean(axis=0))
sample_means = np.stack(sample_means, axis=0)

# 按情绪分组均值
grouped = sample_index.assign(_idx=np.arange(len(sample_index))).groupby("emotion")['_idx'].apply(list)
emotion_names = sorted(grouped.index.tolist())
raw_vectors = np.stack([
    sample_means[grouped[emo]].mean(axis=0)
    for emo in emotion_names
], axis=0)
raw_vectors = raw_vectors - raw_vectors.mean(axis=0, keepdims=True)

# neutral story embeddings 用自身中心化去噪
neutral_embeddings = sample_means[sample_index['emotion'] == 'neutral']
orth_vectors, pc_count = _orthogonalize(raw_vectors, neutral_embeddings, VARIANCE_TARGET)

# 保存
np.save(f"{OUTPUT_DIR}/emotion_vectors_raw.npy", raw_vectors)
np.save(f"{OUTPUT_DIR}/emotion_vectors_orth.npy", orth_vectors)
with open(f"{OUTPUT_DIR}/vector_metadata.json", "w") as f:
    json.dump({
        "emotion_names": emotion_names,
        "pc_count_removed": int(pc_count),
        "variance_target": VARIANCE_TARGET,
        "token_pool_start": int(sample_index.pool_start_token.iloc[0]),
    }, f, indent=2)

print("Done! 已保存 emotion_vectors_raw.npy, emotion_vectors_orth.npy, vector_metadata.json")
