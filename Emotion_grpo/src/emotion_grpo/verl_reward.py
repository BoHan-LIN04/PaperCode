from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from emotion_grpo.rewards.registry import build_provider


def build_reward_inputs(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    extra_info = dict(extra_info or {})
    metadata = dict(extra_info.get("metadata") or {})
    metadata.setdefault("data_source", data_source)
    metadata.setdefault("ground_truth", ground_truth)

    batch_record = {
        "messages": extra_info.get("messages", []),
        "metadata": metadata,
        "data_source": data_source,
        "ground_truth": ground_truth,
    }
    return [batch_record], [solution_str], [metadata]


@lru_cache(maxsize=16)
def _get_cached_provider(provider_cls: str, provider_kwargs_json: str):
    kwargs = json.loads(provider_kwargs_json)
    return build_provider(provider_cls=provider_cls, provider_kwargs=kwargs)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    provider_cls: str = "emotion_grpo.rewards.random_provider.RandomIntrinsicRewardProvider",
    provider_kwargs: dict[str, Any] | None = None,
    include_details: bool = True,
    **_: Any,
) -> float | dict[str, Any]:
    provider_kwargs = provider_kwargs or {}
    batch_records, generations, metadata = build_reward_inputs(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    provider = _get_cached_provider(provider_cls, json.dumps(provider_kwargs, sort_keys=True))
    score = float(provider.score_batch(batch_records, generations, metadata)[0])

    if not include_details:
        return score

    return {
        "score": score,
        "provider": provider.__class__.__name__,
        "data_source": data_source,
        "response_preview": solution_str[:120],
    }

