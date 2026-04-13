from __future__ import annotations

import hashlib
import json
from typing import Any

from emotion_grpo.rewards.base import IntrinsicRewardProvider


class RandomIntrinsicRewardProvider(IntrinsicRewardProvider):
    def __init__(self, seed: int = 7, min_value: float = -1.0, max_value: float = 1.0) -> None:
        if max_value < min_value:
            raise ValueError("max_value must be greater than or equal to min_value")
        self.seed = seed
        self.min_value = min_value
        self.max_value = max_value

    def _score_one(self, record: dict[str, Any], generation: str, metadata: dict[str, Any]) -> float:
        payload = {
            "seed": self.seed,
            "record": record,
            "generation": generation,
            "metadata": metadata,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).digest()
        numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
        fraction = numerator / float((1 << 64) - 1)
        return self.min_value + (self.max_value - self.min_value) * fraction

    def score_batch(
        self,
        batch_records: list[dict[str, Any]],
        generations: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[float]:
        if not (len(batch_records) == len(generations) == len(metadata)):
            raise ValueError("batch_records, generations, and metadata must have the same length")
        return [
            self._score_one(record=record, generation=generation, metadata=meta)
            for record, generation, meta in zip(batch_records, generations, metadata)
        ]

