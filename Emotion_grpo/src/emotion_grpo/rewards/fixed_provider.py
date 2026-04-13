from __future__ import annotations

from typing import Any

from emotion_grpo.rewards.base import IntrinsicRewardProvider


class FixedIntrinsicRewardProvider(IntrinsicRewardProvider):
    def __init__(self, default_value: float = 0.0, metadata_field: str | None = None) -> None:
        self.default_value = default_value
        self.metadata_field = metadata_field

    def score_batch(
        self,
        batch_records: list[dict[str, Any]],
        generations: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[float]:
        del batch_records, generations
        scores: list[float] = []
        for meta in metadata:
            if self.metadata_field and self.metadata_field in meta:
                scores.append(float(meta[self.metadata_field]))
            else:
                scores.append(float(self.default_value))
        return scores

