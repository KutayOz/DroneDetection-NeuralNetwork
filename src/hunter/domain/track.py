"""
Track domain entities.

Contains business logic for track identity and state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TrackState(Enum):
    """
    Track lifecycle states (Eagle Model).

    Transitions:
        TENTATIVE -> CONFIRMED (after N consecutive hits)
        CONFIRMED -> LOST (after M consecutive misses)
        LOST -> CONFIRMED (if re-associated)
        LOST -> DELETED (after timeout)
        TENTATIVE -> DELETED (if not confirmed in time)
    """

    TENTATIVE = auto()   # New track, awaiting confirmation
    CONFIRMED = auto()   # Stable tracking
    LOST = auto()        # Temporarily lost
    DELETED = auto()     # Track terminated

    def is_active(self) -> bool:
        """Whether track should be updated."""
        return self in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    def is_visible(self) -> bool:
        """Whether track should appear in output."""
        return self in (TrackState.CONFIRMED, TrackState.LOST)


@dataclass(frozen=True)
class TrackIdentity:
    """
    Value object for track identity.

    Immutable identifier that persists through track lifecycle.
    """

    track_id: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrackIdentity):
            return NotImplemented
        return self.track_id == other.track_id

    def __hash__(self) -> int:
        return hash(self.track_id)

    def __str__(self) -> str:
        return f"Track-{self.track_id}"


@dataclass
class TrackMetrics:
    """
    Track statistics and metrics.

    Mutable - updated as track progresses.
    """

    age: int = 0
    hits: int = 0
    misses: int = 0
    time_since_update: int = 0
    _consecutive_hits: int = field(default=0, repr=False)
    _consecutive_misses: int = field(default=0, repr=False)

    def record_hit(self) -> None:
        """Record successful association."""
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        self._consecutive_hits += 1
        self._consecutive_misses = 0

    def record_miss(self) -> None:
        """Record missed association."""
        self.age += 1
        self.misses += 1
        self.time_since_update += 1
        self._consecutive_misses += 1
        self._consecutive_hits = 0

    @property
    def hit_ratio(self) -> float:
        """Ratio of hits to total age."""
        if self.age == 0:
            return 0.0
        return self.hits / self.age

    @property
    def consecutive_hits(self) -> int:
        """Current consecutive hit streak."""
        return self._consecutive_hits

    @property
    def consecutive_misses(self) -> int:
        """Current consecutive miss streak."""
        return self._consecutive_misses

    def reset_streaks(self) -> None:
        """Reset consecutive counters."""
        self._consecutive_hits = 0
        self._consecutive_misses = 0


@dataclass
class TrackTransition:
    """
    Record of a state transition.

    Immutable record for logging/debugging.
    """

    track_id: int
    from_state: TrackState
    to_state: TrackState
    reason: str
    frame_id: int
    timestamp_ms: int

    @property
    def is_creation(self) -> bool:
        """Whether this is track creation."""
        return self.from_state is None or self.from_state == TrackState.TENTATIVE

    @property
    def is_deletion(self) -> bool:
        """Whether this is track deletion."""
        return self.to_state == TrackState.DELETED

    @property
    def is_recovery(self) -> bool:
        """Whether this is recovery from lost."""
        return (
            self.from_state == TrackState.LOST
            and self.to_state == TrackState.CONFIRMED
        )
