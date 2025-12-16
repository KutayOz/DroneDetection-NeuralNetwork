"""
Tracking subsystem.

Provides multi-target tracking with:
- State machine based lifecycle (Eagle Model)
- Kalman filter for motion prediction
- Hungarian algorithm for association
- Trajectory management
"""

from .association import (
    AssociationConfig,
    AssociationResult,
    Detection,
    HungarianAssociator,
    IoUOnlyAssociator,
    compute_cost_matrix,
    compute_embedding_distance,
)
from .kalman_filter import BboxKalmanFilter, KalmanConfig
from .state_machine import StateMachineConfig, TrackState, TrackStateMachine
from .track import Track, TrackConfig
from .tracker import MultiTargetTracker
from .trajectory import TrajectoryManager, TrajectoryPoint

__all__ = [
    # State Machine
    "TrackState",
    "StateMachineConfig",
    "TrackStateMachine",
    # Kalman Filter
    "KalmanConfig",
    "BboxKalmanFilter",
    # Trajectory
    "TrajectoryPoint",
    "TrajectoryManager",
    # Association
    "Detection",
    "AssociationConfig",
    "AssociationResult",
    "HungarianAssociator",
    "IoUOnlyAssociator",
    "compute_cost_matrix",
    "compute_embedding_distance",
    # Track
    "TrackConfig",
    "Track",
    # Tracker
    "MultiTargetTracker",
]
