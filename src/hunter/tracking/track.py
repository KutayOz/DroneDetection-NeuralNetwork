"""
Track class combining all tracking components.

Each Track represents a single tracked target across frames.

Follows Composition over Inheritance: Track composes multiple components.
Follows SRP: Track coordinates components but delegates specific logic.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .state_machine import TrackStateMachine, TrackState, StateMachineConfig
from .kalman_filter import BboxKalmanFilter, KalmanConfig
from .trajectory import TrajectoryManager


@dataclass
class TrackConfig:
    """
    Combined configuration for Track.

    Aggregates configs for all sub-components.
    """

    state_config: StateMachineConfig
    kalman_config: KalmanConfig
    trajectory_max_length: int = 150
    embedding_ema_alpha: float = 0.3  # EMA for embedding updates


class Track:
    """
    Single tracked target.

    Combines:
    - State machine (lifecycle management)
    - Kalman filter (motion prediction)
    - Trajectory (position history)
    - Embedding (appearance features)

    Each track has a unique ID assigned at creation.
    """

    # Class-level ID counter for unique IDs
    _id_counter: int = 0

    def __init__(
        self,
        initial_bbox: Tuple[float, float, float, float],
        initial_confidence: float,
        timestamp_ms: int,
        config: TrackConfig,
        embedding: Optional[np.ndarray] = None,
    ):
        """
        Create a new track.

        Args:
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            initial_confidence: Initial detection confidence
            timestamp_ms: Frame timestamp in milliseconds
            config: Track configuration
            embedding: Optional initial embedding
        """
        # Assign unique ID
        self.track_id = Track._id_counter
        Track._id_counter += 1

        self._config = config

        # Initialize components
        self._state_machine = TrackStateMachine(config.state_config)
        self._kalman = BboxKalmanFilter(config.kalman_config)
        self._kalman.initialize(initial_bbox)
        self._trajectory = TrajectoryManager(config.trajectory_max_length)

        # State
        self._bbox = initial_bbox
        self._predicted_bbox = initial_bbox
        self._confidence = initial_confidence
        self._embedding = embedding.copy() if embedding is not None else None
        self._last_update_ms = timestamp_ms
        self._creation_ms = timestamp_ms
        self._age_frames = 0
        self._hits = 1  # Number of successful matches
        self._time_since_update = 0  # Frames since last detection match

        # Add initial trajectory point
        cx = (initial_bbox[0] + initial_bbox[2]) / 2
        cy = (initial_bbox[1] + initial_bbox[3]) / 2
        self._trajectory.add_point(timestamp_ms, cx, cy)

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> TrackState:
        """Current track state."""
        return self._state_machine.state

    @property
    def state_name(self) -> str:
        """Current state name."""
        return self._state_machine.state_name

    @property
    def is_active(self) -> bool:
        """Whether track is still active (not dropped)."""
        return self._state_machine.is_active

    @property
    def is_visible(self) -> bool:
        """Whether track should appear in output."""
        return self._state_machine.is_visible

    @property
    def is_confirmed(self) -> bool:
        """Whether track has been confirmed at least once."""
        return self._state_machine.is_confirmed

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Current bounding box (x1, y1, x2, y2)."""
        return self._bbox

    @property
    def bbox_int(self) -> Tuple[int, int, int, int]:
        """Current bounding box as integers."""
        return (
            int(self._bbox[0]),
            int(self._bbox[1]),
            int(self._bbox[2]),
            int(self._bbox[3]),
        )

    @property
    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        """Predicted bounding box from Kalman filter."""
        return self._predicted_bbox

    @property
    def confidence(self) -> float:
        """Latest detection confidence."""
        return self._confidence

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Current appearance embedding (smoothed with EMA)."""
        return self._embedding

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity (vx, vy) in pixels per frame."""
        return self._kalman.velocity

    @property
    def velocity_per_second(self) -> Tuple[float, float]:
        """Velocity (vx, vy) in pixels per second."""
        return self._kalman.velocity_per_second

    @property
    def center(self) -> Tuple[float, float]:
        """Current center position (cx, cy)."""
        return self._kalman.center

    @property
    def age_frames(self) -> int:
        """Track age in frames."""
        return self._age_frames

    @property
    def hits(self) -> int:
        """Number of successful detection matches."""
        return self._hits

    @property
    def time_since_update(self) -> int:
        """Frames since last detection match."""
        return self._time_since_update

    @property
    def match_ratio(self) -> float:
        """Ratio of hits to age."""
        return self._hits / self._age_frames if self._age_frames > 0 else 1.0

    @property
    def trajectory_length(self) -> int:
        """Number of points in trajectory."""
        return self._trajectory.length

    @property
    def last_update_ms(self) -> int:
        """Timestamp of last update."""
        return self._last_update_ms

    @property
    def creation_ms(self) -> int:
        """Timestamp of track creation."""
        return self._creation_ms

    # ─────────────────────────────────────────────────────────────────────
    # Methods
    # ─────────────────────────────────────────────────────────────────────

    def predict(self) -> Tuple[float, float, float, float]:
        """
        Run Kalman prediction step.

        Should be called at the start of each frame,
        before association.

        Returns:
            Predicted bbox (x1, y1, x2, y2)
        """
        self._predicted_bbox = self._kalman.predict()
        self._age_frames += 1
        self._time_since_update += 1
        return self._predicted_bbox

    def update(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        confidence: Optional[float],
        timestamp_ms: int,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[TrackState, Optional[str]]:
        """
        Update track with detection (or mark as missed).

        Should be called after association, once per frame.

        Args:
            bbox: Matched detection bbox (None if missed)
            confidence: Detection confidence (None if missed)
            timestamp_ms: Current frame timestamp
            embedding: Detection embedding (optional)

        Returns:
            Tuple of (new_state, transition_reason)
        """
        matched = bbox is not None

        if matched:
            # Update Kalman with measurement
            self._bbox = self._kalman.update(bbox)
            self._confidence = confidence
            self._hits += 1
            self._time_since_update = 0

            # Update embedding with exponential moving average
            if embedding is not None:
                self._update_embedding(embedding)

            # Add to trajectory
            cx = (self._bbox[0] + self._bbox[2]) / 2
            cy = (self._bbox[1] + self._bbox[3]) / 2
            self._trajectory.add_point(timestamp_ms, cx, cy)

        self._last_update_ms = timestamp_ms

        # Update state machine
        return self._state_machine.update(matched)

    def _update_embedding(self, new_embedding: np.ndarray) -> None:
        """
        Update embedding using exponential moving average.

        Args:
            new_embedding: New embedding from detection
        """
        if self._embedding is None:
            self._embedding = new_embedding.copy()
        else:
            alpha = self._config.embedding_ema_alpha
            self._embedding = alpha * new_embedding + (1 - alpha) * self._embedding

            # Re-normalize
            norm = np.linalg.norm(self._embedding)
            if norm > 0:
                self._embedding = self._embedding / norm

    def get_trajectory_tail(self, n: int = 10) -> List[dict]:
        """
        Get last N trajectory points.

        Args:
            n: Number of points

        Returns:
            List of point dictionaries
        """
        return self._trajectory.get_tail(n)

    def get_predicted_position(self, steps: int = 1) -> Tuple[float, float]:
        """
        Predict future center position.

        Args:
            steps: Number of time steps ahead

        Returns:
            Predicted (cx, cy)
        """
        return self._kalman.get_predicted_position(steps)

    def force_drop(self, reason: str = "forced") -> str:
        """
        Force track to drop.

        Args:
            reason: Reason for dropping

        Returns:
            The reason string
        """
        return self._state_machine.force_drop(reason)

    def compute_similarity(self, other_embedding: np.ndarray) -> float:
        """
        Compute embedding similarity with another embedding.

        Args:
            other_embedding: Embedding to compare with

        Returns:
            Cosine similarity (-1 to 1)
        """
        if self._embedding is None:
            return 0.0
        return float(np.dot(self._embedding, other_embedding))

    # ─────────────────────────────────────────────────────────────────────
    # Class Methods
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset ID counter to 0 (for testing)."""
        cls._id_counter = 0

    @classmethod
    def get_next_id(cls) -> int:
        """Get the next ID that will be assigned."""
        return cls._id_counter

    # ─────────────────────────────────────────────────────────────────────
    # Magic Methods
    # ─────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Track(id={self.track_id}, state={self.state_name}, "
            f"pos=({cx:.1f}, {cy:.1f}), age={self._age_frames})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Track):
            return self.track_id == other.track_id
        return False

    def __hash__(self) -> int:
        return hash(self.track_id)
