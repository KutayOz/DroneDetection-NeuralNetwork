"""
Trajectory management for tracks.

Stores history of track positions for:
- Visualization
- Motion analysis
- Trajectory prediction

Follows SRP: Only responsible for trajectory storage and retrieval.
"""

from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    Single point in a trajectory.

    Immutable for thread safety.
    """

    timestamp_ms: int
    cx: float  # Center x
    cy: float  # Center y

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "t_ms": self.timestamp_ms,
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
        }


class TrajectoryManager:
    """
    Manages trajectory history for a single track.

    Uses a ring buffer (deque) for efficient storage.
    """

    def __init__(self, max_length: int = 150):
        """
        Initialize trajectory manager.

        Args:
            max_length: Maximum number of points to store
        """
        if max_length < 1:
            raise ValueError("max_length must be >= 1")

        self._max_length = max_length
        self._points: deque[TrajectoryPoint] = deque(maxlen=max_length)

    def add_point(self, timestamp_ms: int, cx: float, cy: float) -> None:
        """
        Add a new point to the trajectory.

        Args:
            timestamp_ms: Timestamp in milliseconds
            cx: Center x coordinate
            cy: Center y coordinate
        """
        point = TrajectoryPoint(timestamp_ms=timestamp_ms, cx=cx, cy=cy)
        self._points.append(point)

    def add_from_bbox(
        self, timestamp_ms: int, bbox_xyxy: tuple[float, float, float, float]
    ) -> None:
        """
        Add a point from bbox coordinates.

        Args:
            timestamp_ms: Timestamp in milliseconds
            bbox_xyxy: Bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.add_point(timestamp_ms, cx, cy)

    def get_tail(self, n: int = 10) -> List[dict]:
        """
        Get last N points as dictionaries.

        Args:
            n: Number of points to return

        Returns:
            List of point dictionaries
        """
        points = list(self._points)[-n:]
        return [p.to_dict() for p in points]

    def get_points(self, n: Optional[int] = None) -> List[TrajectoryPoint]:
        """
        Get trajectory points.

        Args:
            n: Number of points (None = all)

        Returns:
            List of TrajectoryPoint objects
        """
        if n is None:
            return list(self._points)
        return list(self._points)[-n:]

    def get_latest(self) -> Optional[TrajectoryPoint]:
        """Get the most recent point."""
        return self._points[-1] if self._points else None

    def get_oldest(self) -> Optional[TrajectoryPoint]:
        """Get the oldest point."""
        return self._points[0] if self._points else None

    @property
    def length(self) -> int:
        """Number of points in trajectory."""
        return len(self._points)

    @property
    def is_empty(self) -> bool:
        """Whether trajectory is empty."""
        return len(self._points) == 0

    @property
    def duration_ms(self) -> int:
        """Duration from oldest to newest point in milliseconds."""
        if len(self._points) < 2:
            return 0
        return self._points[-1].timestamp_ms - self._points[0].timestamp_ms

    @property
    def average_velocity(self) -> tuple[float, float]:
        """
        Calculate average velocity over trajectory.

        Returns:
            Tuple of (vx, vy) in pixels per millisecond
        """
        if len(self._points) < 2:
            return (0.0, 0.0)

        oldest = self._points[0]
        newest = self._points[-1]

        dt = newest.timestamp_ms - oldest.timestamp_ms
        if dt <= 0:
            return (0.0, 0.0)

        vx = (newest.cx - oldest.cx) / dt
        vy = (newest.cy - oldest.cy) / dt

        return (vx, vy)

    def get_displacement(self) -> tuple[float, float]:
        """
        Get total displacement from start to end.

        Returns:
            Tuple of (dx, dy) in pixels
        """
        if len(self._points) < 2:
            return (0.0, 0.0)

        oldest = self._points[0]
        newest = self._points[-1]

        return (newest.cx - oldest.cx, newest.cy - oldest.cy)

    def get_path_length(self) -> float:
        """
        Calculate total path length.

        Returns:
            Total distance traveled in pixels
        """
        if len(self._points) < 2:
            return 0.0

        total = 0.0
        points = list(self._points)

        for i in range(1, len(points)):
            dx = points[i].cx - points[i - 1].cx
            dy = points[i].cy - points[i - 1].cy
            total += (dx**2 + dy**2) ** 0.5

        return total

    def predict_position(self, future_ms: int) -> tuple[float, float]:
        """
        Predict future position using linear extrapolation.

        Args:
            future_ms: Milliseconds into the future

        Returns:
            Predicted (cx, cy) position
        """
        if len(self._points) < 2:
            if self._points:
                return (self._points[-1].cx, self._points[-1].cy)
            return (0.0, 0.0)

        vx, vy = self.average_velocity
        latest = self._points[-1]

        return (
            latest.cx + vx * future_ms,
            latest.cy + vy * future_ms,
        )

    def clear(self) -> None:
        """Clear all trajectory points."""
        self._points.clear()

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        """Iterate over trajectory points."""
        return iter(self._points)

    def __len__(self) -> int:
        """Number of points."""
        return len(self._points)

    def __repr__(self) -> str:
        return f"TrajectoryManager(length={len(self._points)}, duration={self.duration_ms}ms)"
