"""
Tracker interfaces using Python Protocol classes.

Defines contracts for:
- ITracker: Multi-object tracker
- IAssociator: Detection-track association
- IStateMachine: Track lifecycle management
- IMotionModel: Motion prediction (Kalman filter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .detector import BoundingBox, DetectionResult


# ============================================
# Enumerations
# ============================================


class TrackState(Enum):
    """
    Track lifecycle states (Eagle Model).

    State transitions:
        TENTATIVE -> CONFIRMED -> LOST -> DELETED
                  or
        TENTATIVE -> DELETED (if not confirmed in time)
                  or
        LOST -> CONFIRMED (if re-associated)
    """

    TENTATIVE = auto()   # New track, awaiting confirmation
    CONFIRMED = auto()   # Stable tracking
    LOST = auto()        # Temporarily lost, using prediction
    DELETED = auto()     # Track terminated

    def is_active(self) -> bool:
        """Whether track should be updated."""
        return self in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    def is_visible(self) -> bool:
        """Whether track should appear in output."""
        return self in (TrackState.CONFIRMED, TrackState.LOST)


# ============================================
# Data Transfer Objects
# ============================================


@dataclass
class TrajectoryPoint:
    """Single point in trajectory history."""

    timestamp_ms: int
    cx: float
    cy: float
    confidence: float = 1.0


@dataclass
class TrackSnapshot:
    """
    Immutable snapshot of track state for output.

    Contains all information needed for TrackMessage.
    """

    track_id: int
    state: TrackState
    bbox: BoundingBox
    predicted_bbox: BoundingBox
    confidence: float
    velocity: Tuple[float, float]  # (vx, vy) in pixels/second
    age_frames: int
    hits: int
    time_since_update: int
    embedding: Optional[np.ndarray] = None
    trajectory: List[TrajectoryPoint] = field(default_factory=list)

    @property
    def state_name(self) -> str:
        """String representation of state."""
        return self.state.name


@dataclass
class AssociationResult:
    """Result of detection-track association."""

    matched_pairs: List[Tuple[int, int]]  # (track_idx, detection_idx)
    unmatched_tracks: List[int]
    unmatched_detections: List[int]
    cost_matrix: Optional[np.ndarray] = None


# ============================================
# Protocol Interfaces
# ============================================


@runtime_checkable
class IMotionModel(Protocol):
    """
    Interface for motion prediction models.

    Implementations: KalmanMotionModel, ConstantVelocityModel
    """

    def predict(self) -> BoundingBox:
        """
        Predict next state (without measurement).

        Returns:
            Predicted bounding box
        """
        ...

    def update(self, bbox: BoundingBox) -> BoundingBox:
        """
        Update state with measurement.

        Args:
            bbox: Measured bounding box

        Returns:
            Filtered/updated bounding box
        """
        ...

    def initialize(self, bbox: BoundingBox) -> None:
        """
        Initialize state with first measurement.

        Args:
            bbox: Initial bounding box
        """
        ...

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current velocity estimate (vx, vy) in pixels/frame."""
        ...

    @property
    def state_vector(self) -> np.ndarray:
        """Full state vector [cx, cy, w, h, vx, vy, vw, vh]."""
        ...


@runtime_checkable
class IStateMachine(Protocol):
    """
    Interface for track state machine.

    Manages track lifecycle: TENTATIVE -> CONFIRMED -> LOST -> DELETED
    """

    @property
    def state(self) -> TrackState:
        """Current track state."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether track should be updated."""
        ...

    @property
    def is_visible(self) -> bool:
        """Whether track should appear in output."""
        ...

    @property
    def hits(self) -> int:
        """Number of successful associations."""
        ...

    @property
    def age(self) -> int:
        """Total frames since track creation."""
        ...

    @property
    def time_since_update(self) -> int:
        """Frames since last successful association."""
        ...

    def update(self, matched: bool) -> Tuple[TrackState, Optional[str]]:
        """
        Update state machine with association result.

        Args:
            matched: Whether detection was associated this frame

        Returns:
            (new_state, transition_reason) - reason is None if no transition
        """
        ...

    def force_delete(self, reason: str) -> None:
        """Force track deletion."""
        ...


@runtime_checkable
class IAssociator(Protocol):
    """
    Interface for detection-track association.

    Implementations: HungarianAssociator, GreedyAssociator
    """

    def associate(
        self,
        tracks: List[TrackSnapshot],
        detections: List[DetectionResult],
    ) -> AssociationResult:
        """
        Associate detections with tracks.

        Uses cost matrix combining:
        - IoU distance
        - Embedding distance (if available)
        - Motion distance

        Args:
            tracks: Current track snapshots
            detections: Current frame detections

        Returns:
            AssociationResult with matches and unmatched indices
        """
        ...


@runtime_checkable
class ITracker(Protocol):
    """
    Interface for multi-object tracker.

    Orchestrates motion models, state machines, and association.
    """

    def update(
        self,
        detections: List[DetectionResult],
        timestamp_ms: int,
    ) -> List[TrackSnapshot]:
        """
        Update tracker with new detections.

        Pipeline:
        1. Predict all tracks
        2. Associate detections with tracks
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Update state machines
        6. Cleanup deleted tracks

        Args:
            detections: Current frame detections
            timestamp_ms: Frame timestamp

        Returns:
            List of active track snapshots
        """
        ...

    def get_visible_tracks(self) -> List[TrackSnapshot]:
        """Get tracks that should appear in output."""
        ...

    def get_all_tracks(self) -> List[TrackSnapshot]:
        """Get all active tracks (including tentative)."""
        ...

    def reset(self) -> None:
        """Reset tracker state, clear all tracks."""
        ...

    @property
    def track_count(self) -> int:
        """Number of active tracks."""
        ...

    @property
    def visible_track_count(self) -> int:
        """Number of visible tracks."""
        ...

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        ...
