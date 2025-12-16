"""
Track state machine implementation (Eagle Model / Kartal Modeli).

State transitions:
    SEARCH → LOCK → TRACK → LOST → RECOVER → TRACK
                  ↘      ↘           ↘
                  DROP   DROP        DROP

Follows SRP: Only responsible for state management.
Follows OCP: Can add new states without modifying existing logic.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class TrackState(Enum):
    """
    Track lifecycle states.

    Mimics eagle hunting behavior:
    - SEARCH: Scanning for targets
    - LOCK: Target acquired, confirming
    - TRACK: Actively tracking confirmed target
    - LOST: Target temporarily lost
    - RECOVER: Attempting to re-acquire
    - DROPPED: Track terminated
    """

    SEARCH = auto()
    LOCK = auto()
    TRACK = auto()
    LOST = auto()
    RECOVER = auto()
    DROPPED = auto()


@dataclass
class StateMachineConfig:
    """
    Configuration for state transitions.

    All values represent frame counts.
    """

    # LOCK → TRACK: consecutive matches needed
    lock_confirm_frames: int = 3

    # LOCK → DROPPED: max frames without confirmation
    lock_timeout_frames: int = 5

    # TRACK → LOST: consecutive misses
    lost_timeout_frames: int = 30

    # LOST → DROPPED: max frames to attempt recovery
    recover_max_frames: int = 15

    # RECOVER → TRACK: consecutive matches needed
    recover_confirm_frames: int = 2

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.lock_confirm_frames < 1:
            raise ValueError("lock_confirm_frames must be >= 1")
        if self.lock_timeout_frames < self.lock_confirm_frames:
            raise ValueError("lock_timeout_frames must be >= lock_confirm_frames")
        if self.lost_timeout_frames < 1:
            raise ValueError("lost_timeout_frames must be >= 1")
        if self.recover_max_frames < 1:
            raise ValueError("recover_max_frames must be >= 1")


class TrackStateMachine:
    """
    State machine for individual track lifecycle.

    Each Track instance has its own state machine.
    Thread-safe: no shared mutable state.

    State Transitions:
    -----------------
    SEARCH → LOCK: First detection match
    LOCK → TRACK: Confirmed after N consecutive matches
    LOCK → DROPPED: Timeout without confirmation
    TRACK → LOST: No match for M frames
    LOST → RECOVER: Detection matched again
    LOST → DROPPED: Recovery timeout
    RECOVER → TRACK: Confirmed re-acquisition
    RECOVER → LOST: Failed to re-acquire
    """

    def __init__(self, config: Optional[StateMachineConfig] = None):
        """
        Initialize state machine.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self._config = config or StateMachineConfig()
        self._state = TrackState.SEARCH
        self._frames_in_state = 0
        self._consecutive_matches = 0
        self._consecutive_misses = 0
        self._total_matches = 0
        self._total_misses = 0

    @property
    def state(self) -> TrackState:
        """Current state."""
        return self._state

    @property
    def state_name(self) -> str:
        """Current state name as string."""
        return self._state.name

    @property
    def is_active(self) -> bool:
        """Whether track should be kept in memory."""
        return self._state != TrackState.DROPPED

    @property
    def is_visible(self) -> bool:
        """Whether track should appear in output."""
        return self._state in (TrackState.LOCK, TrackState.TRACK, TrackState.RECOVER)

    @property
    def is_confirmed(self) -> bool:
        """Whether track has been confirmed at least once."""
        return self._state in (TrackState.TRACK, TrackState.LOST, TrackState.RECOVER)

    @property
    def frames_in_state(self) -> int:
        """Number of frames in current state."""
        return self._frames_in_state

    @property
    def consecutive_matches(self) -> int:
        """Number of consecutive detection matches."""
        return self._consecutive_matches

    @property
    def consecutive_misses(self) -> int:
        """Number of consecutive detection misses."""
        return self._consecutive_misses

    @property
    def match_ratio(self) -> float:
        """Ratio of matches to total updates."""
        total = self._total_matches + self._total_misses
        return self._total_matches / total if total > 0 else 0.0

    def update(self, matched: bool) -> Tuple[TrackState, Optional[str]]:
        """
        Update state machine with match result.

        This is the core method called every frame for each track.

        Args:
            matched: Whether a detection matched this track

        Returns:
            Tuple of (new_state, transition_reason)
            transition_reason is None if no transition occurred
        """
        old_state = self._state
        reason: Optional[str] = None

        # Update counters
        if matched:
            self._consecutive_matches += 1
            self._consecutive_misses = 0
            self._total_matches += 1
        else:
            self._consecutive_misses += 1
            self._consecutive_matches = 0
            self._total_misses += 1

        self._frames_in_state += 1

        # Apply state transitions
        if self._state == TrackState.SEARCH:
            reason = self._update_search(matched)

        elif self._state == TrackState.LOCK:
            reason = self._update_lock(matched)

        elif self._state == TrackState.TRACK:
            reason = self._update_track(matched)

        elif self._state == TrackState.LOST:
            reason = self._update_lost(matched)

        elif self._state == TrackState.RECOVER:
            reason = self._update_recover(matched)

        # Reset frame counter on state change
        if self._state != old_state:
            self._frames_in_state = 0
            # Reset consecutive counters on state change
            self._consecutive_matches = 0
            self._consecutive_misses = 0

        return self._state, reason

    def _update_search(self, matched: bool) -> Optional[str]:
        """Handle SEARCH state transitions."""
        if matched:
            self._state = TrackState.LOCK
            return "first_match"
        return None

    def _update_lock(self, matched: bool) -> Optional[str]:
        """Handle LOCK state transitions."""
        if self._consecutive_matches >= self._config.lock_confirm_frames:
            self._state = TrackState.TRACK
            return f"confirmed_after_{self._consecutive_matches}_matches"

        if self._consecutive_misses >= self._config.lock_timeout_frames:
            self._state = TrackState.DROPPED
            return "lock_timeout"

        return None

    def _update_track(self, matched: bool) -> Optional[str]:
        """Handle TRACK state transitions."""
        if self._consecutive_misses >= self._config.lost_timeout_frames:
            self._state = TrackState.LOST
            return f"lost_after_{self._consecutive_misses}_misses"
        return None

    def _update_lost(self, matched: bool) -> Optional[str]:
        """Handle LOST state transitions."""
        if matched:
            self._state = TrackState.RECOVER
            return "reacquired"

        if self._frames_in_state >= self._config.recover_max_frames:
            self._state = TrackState.DROPPED
            return "recovery_timeout"

        return None

    def _update_recover(self, matched: bool) -> Optional[str]:
        """Handle RECOVER state transitions."""
        if self._consecutive_matches >= self._config.recover_confirm_frames:
            self._state = TrackState.TRACK
            return "recovered"

        if self._consecutive_misses >= 3:  # Quick fail for recovery
            self._state = TrackState.LOST
            return "recover_failed"

        return None

    def force_drop(self, reason: str = "forced") -> str:
        """
        Forcefully terminate track.

        Args:
            reason: Reason for dropping

        Returns:
            The reason string
        """
        self._state = TrackState.DROPPED
        return reason

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._state = TrackState.SEARCH
        self._frames_in_state = 0
        self._consecutive_matches = 0
        self._consecutive_misses = 0
        self._total_matches = 0
        self._total_misses = 0

    def __repr__(self) -> str:
        return (
            f"TrackStateMachine(state={self._state.name}, "
            f"frames={self._frames_in_state}, "
            f"matches={self._consecutive_matches}, "
            f"misses={self._consecutive_misses})"
        )
