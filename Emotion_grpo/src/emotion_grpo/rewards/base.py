from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IntrinsicRewardProvider(ABC):
    @abstractmethod
    def score_batch(
        self,
        batch_records: list[dict[str, Any]],
        generations: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[float]:
        raise NotImplementedError

