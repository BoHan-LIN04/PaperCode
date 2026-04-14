from __future__ import annotations

import re
from typing import Any

from emotion_grpo.rewards.base import IntrinsicRewardProvider


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_logical_answer(text: Any) -> str:
    out = str(text or "").strip()
    if not out:
        return ""
    boxed = BOXED_RE.findall(out)
    if boxed:
        out = boxed[-1].strip()
    out = CODE_FENCE_RE.sub(" ", out)
    out = out.replace(",", "")
    out = WHITESPACE_RE.sub(" ", out).strip()
    lowered = out.lower()
    for prefix in ("final answer:", "answer:", "the answer is", "final answer is"):
        if lowered.startswith(prefix):
            out = out[len(prefix) :].strip(" :")
            lowered = out.lower()
    number_hits = NUMBER_RE.findall(out)
    if number_hits:
        return number_hits[-1]
    return lowered.strip(" .")


class LogicalExactMatchRewardProvider(IntrinsicRewardProvider):
    def __init__(
        self,
        *,
        target_field: str = "ground_truth",
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
        missing_target_reward: float = -1.0,
    ) -> None:
        self.target_field = target_field
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.missing_target_reward = float(missing_target_reward)

    def _score_one(self, generation: str, meta: dict[str, Any]) -> float:
        target = meta.get(self.target_field, meta.get("ground_truth", meta.get("label", "")))
        normalized_target = normalize_logical_answer(target)
        if not normalized_target:
            return self.missing_target_reward
        normalized_generation = normalize_logical_answer(generation)
        if normalized_generation == normalized_target:
            return self.correct_reward
        return self.incorrect_reward

    def score_batch(
        self,
        batch_records: list[dict[str, Any]],
        generations: list[str],
        metadata: list[dict[str, Any]],
    ) -> list[float]:
        del batch_records
        if not (len(generations) == len(metadata)):
            raise ValueError("generations and metadata must have the same length")
        return [self._score_one(generation, meta) for generation, meta in zip(generations, metadata)]
