"""
Pipeline timing utilities.

Follows SRP: Only responsible for timing measurement.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class StageTimings:
    """
    Pipeline stage timings in milliseconds.

    Immutable data structure holding timing for each pipeline stage.
    """

    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    embed_ms: float = 0.0
    associate_ms: float = 0.0
    kalman_ms: float = 0.0
    output_ms: float = 0.0

    @property
    def total_e2e_ms(self) -> float:
        """Calculate total end-to-end latency."""
        return sum(
            [
                self.decode_ms,
                self.preprocess_ms,
                self.detect_ms,
                self.embed_ms,
                self.associate_ms,
                self.kalman_ms,
                self.output_ms,
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary with rounded values."""
        return {
            "decode_ms": round(self.decode_ms, 2),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "detect_ms": round(self.detect_ms, 2),
            "embed_ms": round(self.embed_ms, 2),
            "associate_ms": round(self.associate_ms, 2),
            "kalman_ms": round(self.kalman_ms, 2),
            "output_ms": round(self.output_ms, 2),
            "total_e2e_ms": round(self.total_e2e_ms, 2),
        }

    def __repr__(self) -> str:
        return f"StageTimings(total={self.total_e2e_ms:.2f}ms)"


class PipelineTimer:
    """
    Context manager based pipeline timing.

    Usage:
        timer = PipelineTimer()

        with timer.measure("detect"):
            result = detector.detect(frame)

        timings = timer.get_timings()
    """

    # Valid stage names that can be measured
    VALID_STAGES = frozenset(
        ["decode", "preprocess", "detect", "embed", "associate", "kalman", "output"]
    )

    def __init__(self):
        self._current = StageTimings()

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        """
        Context manager to measure stage duration.

        Args:
            stage: Name of the stage to measure

        Raises:
            ValueError: If stage name is invalid
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Unknown stage: {stage}. Valid stages: {sorted(self.VALID_STAGES)}"
            )

        attr_name = f"{stage}_ms"
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            setattr(self._current, attr_name, elapsed_ms)

    def reset(self) -> None:
        """Reset all timings to zero."""
        self._current = StageTimings()

    def get_timings(self) -> StageTimings:
        """Get current stage timings."""
        return self._current

    def get_dict(self) -> dict[str, float]:
        """Get timings as dictionary."""
        return self._current.to_dict()


class ScopedTimer:
    """
    Simple scoped timer for ad-hoc timing.

    Usage:
        with ScopedTimer() as t:
            do_something()
        print(f"Took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self):
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "ScopedTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self._end - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._end - self._start
