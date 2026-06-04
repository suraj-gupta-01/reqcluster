from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


def elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds from a perf_counter start value."""
    return round((time.perf_counter() - start) * 1000.0, 3)


@contextmanager
def record_duration_ms(durations: Dict[str, float], stage: str) -> Iterator[None]:
    """Record a stage duration in milliseconds into the provided dictionary."""
    start = time.perf_counter()
    try:
        yield
    finally:
        durations[stage] = elapsed_ms(start)
