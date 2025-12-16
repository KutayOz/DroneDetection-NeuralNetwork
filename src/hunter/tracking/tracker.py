"""
Multi-target tracker orchestrator.

Manages all tracks, performs association, creates/drops tracks.

Follows Facade pattern: Provides simple interface to complex tracking subsystem.
Follows DIP: Depends on abstractions (config), not concrete implementations.
"""

from typing import List, Optional, Tuple
import numpy as np

from .track import Track, TrackConfig
from .association import (
    HungarianAssociator,
    AssociationConfig,
    Detection,
    AssociationResult,
)
from .state_machine import StateMachineConfig, TrackState
from .kalman_filter import KalmanConfig
from ..core.config import TrackingConfig
from ..core.logger import PipelineLogger


class MultiTargetTracker:
    """
    Multi-target tracker.

    Orchestrates the tracking pipeline:
    1. Predict all tracks
    2. Associate detections with tracks
    3. Update matched tracks
    4. Mark unmatched tracks as missed
    5. Create new tracks for unmatched detections
    6. Remove dropped tracks

    Thread-safe: Can be used from multiple threads if detections
    are processed sequentially.
    """

    def __init__(
        self,
        config: TrackingConfig,
        logger: Optional[PipelineLogger] = None,
        fps: float = 30.0,
    ):
        """
        Initialize tracker.

        Args:
            config: Tracking configuration
            logger: Optional pipeline logger
            fps: Video frame rate (for Kalman filter dt)
        """
        self._config = config
        self._logger = logger
        self._fps = fps

        # Build component configurations
        self._track_config = TrackConfig(
            state_config=StateMachineConfig(
                lock_confirm_frames=config.lock_confirm_frames,
                lock_timeout_frames=config.lock_timeout_frames,
                lost_timeout_frames=config.lost_timeout_frames,
                recover_max_frames=config.recover_max_frames,
                recover_confirm_frames=config.recover_confirm_frames,
            ),
            kalman_config=KalmanConfig(
                process_noise=config.process_noise,
                measurement_noise=config.measurement_noise,
                dt=1.0 / fps,
            ),
            trajectory_max_length=config.trajectory_max_length,
        )

        # Initialize associator
        self._associator = HungarianAssociator(
            AssociationConfig(
                iou_threshold=config.iou_threshold,
                embedding_weight=config.embedding_weight,
                gate_threshold=config.gate_threshold,
            )
        )

        # Track storage
        self._tracks: List[Track] = []
        self._frame_count = 0

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def tracks(self) -> List[Track]:
        """All active tracks."""
        return self._tracks

    @property
    def track_count(self) -> int:
        """Number of active tracks."""
        return len(self._tracks)

    @property
    def visible_track_count(self) -> int:
        """Number of visible tracks."""
        return sum(1 for t in self._tracks if t.is_visible)

    @property
    def confirmed_track_count(self) -> int:
        """Number of confirmed tracks."""
        return sum(1 for t in self._tracks if t.is_confirmed)

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count

    # ─────────────────────────────────────────────────────────────────────
    # Main Update Method
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Detection],
        timestamp_ms: int,
    ) -> List[Track]:
        """
        Process one frame of detections.

        This is the main entry point, called once per frame.

        Args:
            detections: Detections from detector
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            List of all active tracks after update
        """
        self._frame_count += 1

        # 1. Predict all active tracks
        active_tracks = [t for t in self._tracks if t.is_active]
        for track in active_tracks:
            track.predict()

        # 2. Associate detections with tracks
        track_boxes = [t.predicted_bbox for t in active_tracks]
        track_embeddings = [t.embedding for t in active_tracks]

        result = self._associator.associate(
            track_boxes,
            track_embeddings,
            detections,
        )

        # 3. Update matched tracks
        self._update_matched_tracks(active_tracks, detections, result, timestamp_ms)

        # 4. Update unmatched tracks (mark as missed)
        self._update_unmatched_tracks(active_tracks, result, timestamp_ms)

        # 5. Create new tracks for unmatched detections
        self._create_new_tracks(detections, result, timestamp_ms)

        # 6. Remove dropped tracks
        self._cleanup_tracks()

        return self._tracks

    def _update_matched_tracks(
        self,
        active_tracks: List[Track],
        detections: List[Detection],
        result: AssociationResult,
        timestamp_ms: int,
    ) -> None:
        """Update tracks that matched with detections."""
        for t_idx, d_idx in result.matched_pairs:
            track = active_tracks[t_idx]
            det = detections[d_idx]

            old_state = track.state
            new_state, reason = track.update(
                bbox=det.bbox_xyxy,
                confidence=det.confidence,
                timestamp_ms=timestamp_ms,
                embedding=det.embedding,
            )

            # Log state change
            if reason and self._logger:
                self._logger.track_state_change(
                    track.track_id,
                    old_state.name,
                    new_state.name,
                    reason,
                )

    def _update_unmatched_tracks(
        self,
        active_tracks: List[Track],
        result: AssociationResult,
        timestamp_ms: int,
    ) -> None:
        """Mark unmatched tracks as missed."""
        for t_idx in result.unmatched_tracks:
            track = active_tracks[t_idx]
            old_state = track.state

            new_state, reason = track.update(
                bbox=None,
                confidence=None,
                timestamp_ms=timestamp_ms,
            )

            # Log state change
            if reason and self._logger:
                self._logger.track_state_change(
                    track.track_id,
                    old_state.name,
                    new_state.name,
                    reason,
                )

    def _create_new_tracks(
        self,
        detections: List[Detection],
        result: AssociationResult,
        timestamp_ms: int,
    ) -> None:
        """Create new tracks for unmatched detections."""
        for d_idx in result.unmatched_detections:
            det = detections[d_idx]

            new_track = Track(
                initial_bbox=det.bbox_xyxy,
                initial_confidence=det.confidence,
                timestamp_ms=timestamp_ms,
                config=self._track_config,
                embedding=det.embedding,
            )

            self._tracks.append(new_track)

            # Log track creation
            if self._logger:
                self._logger.track_created(
                    new_track.track_id,
                    det.bbox_xyxy,
                    det.confidence,
                )

    def _cleanup_tracks(self) -> None:
        """Remove dropped tracks."""
        dropped = [t for t in self._tracks if not t.is_active]

        # Log dropped tracks
        if self._logger:
            for track in dropped:
                self._logger.track_dropped(track.track_id, "state_machine")

        # Keep only active tracks
        self._tracks = [t for t in self._tracks if t.is_active]

    # ─────────────────────────────────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────────────────────────────────

    def get_visible_tracks(self) -> List[Track]:
        """
        Get tracks that should appear in output.

        Returns:
            List of visible tracks (LOCK, TRACK, or RECOVER state)
        """
        return [t for t in self._tracks if t.is_visible]

    def get_confirmed_tracks(self) -> List[Track]:
        """
        Get confirmed tracks.

        Returns:
            List of confirmed tracks (TRACK, LOST, or RECOVER state)
        """
        return [t for t in self._tracks if t.is_confirmed]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.

        Args:
            track_id: Track ID to find

        Returns:
            Track if found, None otherwise
        """
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_tracks_by_state(self, state: TrackState) -> List[Track]:
        """
        Get tracks in specific state.

        Args:
            state: Track state to filter by

        Returns:
            List of tracks in that state
        """
        return [t for t in self._tracks if t.state == state]

    def get_tracks_in_region(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> List[Track]:
        """
        Get tracks with center inside a region.

        Args:
            x1, y1, x2, y2: Region bounds

        Returns:
            List of tracks inside region
        """
        result = []
        for track in self._tracks:
            cx, cy = track.center
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                result.append(track)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Management Methods
    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self._tracks.clear()
        self._frame_count = 0
        Track.reset_id_counter()

    def drop_track(self, track_id: int, reason: str = "manual") -> bool:
        """
        Manually drop a track.

        Args:
            track_id: ID of track to drop
            reason: Reason for dropping

        Returns:
            True if track was found and dropped
        """
        track = self.get_track_by_id(track_id)
        if track:
            track.force_drop(reason)
            if self._logger:
                self._logger.track_dropped(track_id, reason)
            return True
        return False

    def get_stats(self) -> dict:
        """
        Get tracker statistics.

        Returns:
            Dictionary of statistics
        """
        states = {}
        for state in TrackState:
            states[state.name] = len(self.get_tracks_by_state(state))

        return {
            "total_tracks": len(self._tracks),
            "visible_tracks": self.visible_track_count,
            "confirmed_tracks": self.confirmed_track_count,
            "frame_count": self._frame_count,
            "next_track_id": Track.get_next_id(),
            "tracks_by_state": states,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Magic Methods
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of active tracks."""
        return len(self._tracks)

    def __iter__(self):
        """Iterate over tracks."""
        return iter(self._tracks)

    def __repr__(self) -> str:
        return (
            f"MultiTargetTracker(tracks={len(self._tracks)}, "
            f"visible={self.visible_track_count}, "
            f"frames={self._frame_count})"
        )
