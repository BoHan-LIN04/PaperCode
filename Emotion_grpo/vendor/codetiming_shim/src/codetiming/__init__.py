from __future__ import annotations

import logging
import time
from typing import Callable, Optional


class Timer:
    def __init__(
        self,
        name: str | None = None,
        text: str = "{name}: {seconds:.4f}s",
        logger: Optional[Callable[[str], None]] = logging.info,
    ) -> None:
        self.name = name or "Timer"
        self.text = text
        self.logger = logger
        self.start_time: float | None = None
        self.last: float = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")
        self.last = time.perf_counter() - self.start_time
        if self.logger is not None:
            self.logger(self.text.format(name=self.name, seconds=self.last))
        self.start_time = None
        return self.last

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.stop()
        return False
