from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CachePaths:
    root: Path

    @property
    def activations(self) -> Path:
        return self.root / "activations.f16.mmap"

    @property
    def sample_ids(self) -> Path:
        return self.root / "sample_ids.i32.mmap"

    @property
    def token_ids(self) -> Path:
        return self.root / "token_ids.i32.mmap"

    @property
    def token_positions(self) -> Path:
        return self.root / "token_positions.i32.mmap"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"


class ActivationCacheWriter:
    def __init__(self, root: str | Path, num_tokens: int, hidden_size: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = CachePaths(self.root)
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.offset = 0
        self._acts = np.memmap(self.paths.activations, dtype=np.float16, mode="w+", shape=(self.num_tokens, self.hidden_size))
        self._sample_ids = np.memmap(self.paths.sample_ids, dtype=np.int32, mode="w+", shape=(self.num_tokens,))
        self._token_ids = np.memmap(self.paths.token_ids, dtype=np.int32, mode="w+", shape=(self.num_tokens,))
        self._positions = np.memmap(self.paths.token_positions, dtype=np.int32, mode="w+", shape=(self.num_tokens,))

    def write_batch(
        self,
        activations: np.ndarray,
        sample_ids: np.ndarray,
        token_ids: np.ndarray,
        token_positions: np.ndarray,
    ) -> None:
        n = int(activations.shape[0])
        start = self.offset
        end = start + n
        if end > self.num_tokens:
            raise RuntimeError(f"cache overflow: {end} > {self.num_tokens}")
        self._acts[start:end] = activations.astype(np.float16, copy=False)
        self._sample_ids[start:end] = sample_ids.astype(np.int32, copy=False)
        self._token_ids[start:end] = token_ids.astype(np.int32, copy=False)
        self._positions[start:end] = token_positions.astype(np.int32, copy=False)
        self.offset = end

    def flush(self) -> None:
        self._acts.flush()
        self._sample_ids.flush()
        self._token_ids.flush()
        self._positions.flush()

    def write_metadata(self, payload: dict) -> None:
        out = dict(payload)
        out.update({"num_tokens": self.num_tokens, "hidden_size": self.hidden_size, "written_tokens": self.offset})
        with self.paths.metadata.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


class ActivationCacheReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = CachePaths(self.root)
        with self.paths.metadata.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        n = int(self.meta["written_tokens"])
        d = int(self.meta["hidden_size"])
        self.activations = np.memmap(self.paths.activations, dtype=np.float16, mode="r", shape=(n, d))
        self.sample_ids = np.memmap(self.paths.sample_ids, dtype=np.int32, mode="r", shape=(n,))
        self.token_ids = np.memmap(self.paths.token_ids, dtype=np.int32, mode="r", shape=(n,))
        self.token_positions = np.memmap(self.paths.token_positions, dtype=np.int32, mode="r", shape=(n,))
