"""
Runtime metrics collection.

Follows SRP: Each metrics class tracks one type of measurement.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatencyMetrics:
    """
    Sliding window latency statistics.

    Tracks p50, p95, p99 percentiles over a configurable window.
    """

    window_size: int = 1000
    _samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self):
        """Initialize deque with correct maxlen."""
        self._samples = deque(maxlen=self.window_size)

    def add_sample(self, latency_ms: float) -> None:
        """
        Add a latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._samples.append(latency_ms)

    def _percentile(self, p: float) -> float:
        """
        Calculate percentile.

        Args:
            p: Percentile (0.0 to 1.0)

        Returns:
            Percentile value or 0.0 if no samples
        """
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return self._percentile(0.50)

    @property
    def p95(self) -> float:
        """95th percentile."""
        return self._percentile(0.95)

    @property
    def p99(self) -> float:
        """99th percentile."""
        return self._percentile(0.99)

    @property
    def mean(self) -> float:
        """Average latency."""
        return statistics.mean(self._samples) if self._samples else 0.0

    @property
    def min(self) -> float:
        """Minimum latency."""
        return min(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        """Maximum latency."""
        return max(self._samples) if self._samples else 0.0

    @property
    def count(self) -> int:
        """Number of samples."""
        return len(self._samples)

    def reset(self) -> None:
        """Clear all samples."""
        self._samples.clear()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "mean_ms": round(self.mean, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "sample_count": self.count,
        }


@dataclass
class ThroughputMetrics:
    """
    FPS (frames per second) tracking.

    Uses sliding window to calculate current throughput.
    """

    window_seconds: float = 5.0
    _timestamps: deque = field(default_factory=deque)

    def tick(self, timestamp: Optional[float] = None) -> None:
        """
        Record a frame processing timestamp.

        Args:
            timestamp: Unix timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        self._timestamps.append(timestamp)

        # Remove old timestamps outside window
        cutoff = timestamp - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def fps(self) -> float:
        """
        Current frames per second.

        Returns:
            FPS calculated over the sliding window
        """
        if len(self._timestamps) < 2:
            return 0.0

        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / duration

    @property
    def frame_count(self) -> int:
        """Number of frames in current window."""
        return len(self._timestamps)

    def reset(self) -> None:
        """Clear all timestamps."""
        self._timestamps.clear()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "window_seconds": self.window_seconds,
        }


@dataclass
class TrackingMetrics:
    """
    Tracking-specific metrics.

    Tracks statistics about track lifecycle.
    """

    total_tracks_created: int = 0
    total_tracks_dropped: int = 0
    total_id_switches: int = 0
    current_active_tracks: int = 0
    max_concurrent_tracks: int = 0

    def track_created(self) -> None:
        """Record a new track creation."""
        self.total_tracks_created += 1
        self.current_active_tracks += 1
        self.max_concurrent_tracks = max(
            self.max_concurrent_tracks, self.current_active_tracks
        )

    def track_dropped(self) -> None:
        """Record a track being dropped."""
        self.total_tracks_dropped += 1
        self.current_active_tracks = max(0, self.current_active_tracks - 1)

    def id_switch(self) -> None:
        """Record an ID switch event."""
        self.total_id_switches += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_tracks_created = 0
        self.total_tracks_dropped = 0
        self.total_id_switches = 0
        self.current_active_tracks = 0
        self.max_concurrent_tracks = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_tracks_created": self.total_tracks_created,
            "total_tracks_dropped": self.total_tracks_dropped,
            "total_id_switches": self.total_id_switches,
            "current_active_tracks": self.current_active_tracks,
            "max_concurrent_tracks": self.max_concurrent_tracks,
        }


class MetricsCollector:
    """
    Aggregates all metrics.

    Single access point for all runtime metrics.
    Follows Facade pattern for simplified interface.
    """

    def __init__(self, latency_window: int = 1000, throughput_window: float = 5.0):
        self.latency = LatencyMetrics(window_size=latency_window)
        self.throughput = ThroughputMetrics(window_seconds=throughput_window)
        self.tracking = TrackingMetrics()

    def reset_all(self) -> None:
        """Reset all metrics."""
        self.latency.reset()
        self.throughput.reset()
        self.tracking.reset()

    def to_dict(self) -> dict:
        """Get all metrics as dictionary."""
        return {
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "tracking": self.tracking.to_dict(),
        }
