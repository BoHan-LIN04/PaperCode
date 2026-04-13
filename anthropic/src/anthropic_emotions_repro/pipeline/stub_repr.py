from __future__ import annotations

import hashlib

import numpy as np


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def stub_tokenize(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def stub_token_activations(text: str, hidden_size: int) -> tuple[np.ndarray, list[str]]:
    tokens = stub_tokenize(text)
    if not tokens:
        return np.zeros((1, hidden_size), dtype=np.float32), [""]
    rows = []
    for idx, token in enumerate(tokens):
        rng = np.random.default_rng(_seed_from_text(f"{idx}:{token.lower()}"))
        rows.append(rng.standard_normal(hidden_size, dtype=np.float32))
    return np.stack(rows, axis=0), tokens


def stub_text_embedding(text: str, hidden_size: int) -> np.ndarray:
    acts, _ = stub_token_activations(text, hidden_size)
    return acts.mean(axis=0)
