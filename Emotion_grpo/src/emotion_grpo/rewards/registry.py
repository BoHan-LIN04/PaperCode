from __future__ import annotations

import importlib
from typing import Any

from emotion_grpo.rewards.base import IntrinsicRewardProvider


def load_provider_class(provider_cls: str) -> type[IntrinsicRewardProvider]:
    module_name, class_name = provider_cls.rsplit(".", 1)
    module = importlib.import_module(module_name)
    provider_type = getattr(module, class_name)
    if not issubclass(provider_type, IntrinsicRewardProvider):
        raise TypeError(f"{provider_cls} is not an IntrinsicRewardProvider")
    return provider_type


def build_provider(provider_cls: str, provider_kwargs: dict[str, Any] | None = None) -> IntrinsicRewardProvider:
    provider_type = load_provider_class(provider_cls)
    return provider_type(**(provider_kwargs or {}))

