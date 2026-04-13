from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class ResidualVectorIntervener:
    def __init__(self, layer_module, vector: torch.Tensor, strength: float) -> None:
        self.layer_module = layer_module
        self.vector = vector.detach().cpu()
        self.strength = float(strength)
        self._cached = None

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hidden.device or self._cached.dtype != hidden.dtype:
            self._cached = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        delta = self._cached * self.strength
        while delta.ndim < hidden.ndim:
            delta = delta.unsqueeze(0)
        hidden_mod = hidden + delta
        if isinstance(output, tuple):
            return (hidden_mod, *output[1:])
        return hidden_mod

    @contextmanager
    def apply(self) -> Iterator[None]:
        handle = self.layer_module.register_forward_hook(self._hook)
        try:
            yield
        finally:
            handle.remove()
