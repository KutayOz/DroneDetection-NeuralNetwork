"""
Unit tests for TrackStateMachine.
"""

import pytest

from hunter.tracking.state_machine import (
    TrackStateMachine,
    TrackState,
    StateMachineConfig,
)


class TestTrackStateMachine:
    """Tests for TrackStateMachine."""

    def test_initial_state_is_search(self):
        """State machine should start in SEARCH state."""
        sm = TrackStateMachine()
        assert sm.state == TrackState.SEARCH
        assert sm.state_name == "SEARCH"

    def test_search_to_lock_on_match(self):
        """First match should transition to LOCK."""
        sm = TrackStateMachine()
        new_state, reason = sm.update(matched=True)

        assert new_state == TrackState.LOCK
        assert reason == "first_match"

    def test_lock_to_track_after_confirmations(self):
        """Should transition to TRACK after enough confirmations."""
        config = StateMachineConfig(lock_confirm_frames=3)
        sm = TrackStateMachine(config)

        # First match → LOCK
        sm.update(matched=True)
        assert sm.state == TrackState.LOCK

        # More matches to confirm
        sm.update(matched=True)
        sm.update(matched=True)
        new_state, reason = sm.update(matched=True)

        assert new_state == TrackState.TRACK
        assert "confirmed" in reason

    def test_lock_to_dropped_on_timeout(self):
        """Should drop if not confirmed within timeout."""
        config = StateMachineConfig(
            lock_confirm_frames=3,
            lock_timeout_frames=3,  # Must be >= lock_confirm_frames
        )
        sm = TrackStateMachine(config)

        # First match → LOCK
        sm.update(matched=True)

        # Three misses → timeout
        sm.update(matched=False)
        sm.update(matched=False)
        new_state, reason = sm.update(matched=False)

        assert new_state == TrackState.DROPPED
        assert reason == "lock_timeout"

    def test_track_to_lost_on_misses(self):
        """Should transition to LOST after consecutive misses."""
        config = StateMachineConfig(
            lock_confirm_frames=1,
            lost_timeout_frames=3,
        )
        sm = TrackStateMachine(config)

        # Get to TRACK state
        sm.update(matched=True)  # SEARCH → LOCK
        sm.update(matched=True)  # LOCK → TRACK

        assert sm.state == TrackState.TRACK

        # Three misses → LOST
        sm.update(matched=False)
        sm.update(matched=False)
        new_state, reason = sm.update(matched=False)

        assert new_state == TrackState.LOST
        assert "lost" in reason

    def test_lost_to_recover_on_match(self):
        """Should transition to RECOVER when detection found again."""
        config = StateMachineConfig(
            lock_confirm_frames=1,
            lost_timeout_frames=1,
        )
        sm = TrackStateMachine(config)

        # Get to LOST state
        sm.update(matched=True)  # SEARCH → LOCK
        sm.update(matched=True)  # LOCK → TRACK
        sm.update(matched=False)  # TRACK → LOST

        assert sm.state == TrackState.LOST

        # Match again → RECOVER
        new_state, reason = sm.update(matched=True)

        assert new_state == TrackState.RECOVER
        assert reason == "reacquired"

    def test_recover_to_track_on_confirmation(self):
        """Should return to TRACK after recovery confirmation."""
        config = StateMachineConfig(
            lock_confirm_frames=1,
            lost_timeout_frames=1,
            recover_confirm_frames=2,
        )
        sm = TrackStateMachine(config)

        # Get to RECOVER state
        sm.update(matched=True)   # SEARCH → LOCK
        sm.update(matched=True)   # LOCK → TRACK
        sm.update(matched=False)  # TRACK → LOST
        sm.update(matched=True)   # LOST → RECOVER

        assert sm.state == TrackState.RECOVER

        # Confirm recovery
        sm.update(matched=True)
        new_state, reason = sm.update(matched=True)

        assert new_state == TrackState.TRACK
        assert reason == "recovered"

    def test_is_active_property(self):
        """is_active should be False only for DROPPED."""
        sm = TrackStateMachine()

        # Active in SEARCH
        assert sm.is_active

        # Still active in LOCK
        sm.update(matched=True)
        assert sm.is_active

        # Force drop
        sm.force_drop()
        assert not sm.is_active

    def test_is_visible_property(self):
        """is_visible should be True for LOCK, TRACK, RECOVER."""
        config = StateMachineConfig(
            lock_confirm_frames=1,
            lost_timeout_frames=1,
        )
        sm = TrackStateMachine(config)

        # Not visible in SEARCH
        assert not sm.is_visible

        # Visible in LOCK
        sm.update(matched=True)
        assert sm.is_visible
        assert sm.state == TrackState.LOCK

        # Visible in TRACK
        sm.update(matched=True)
        assert sm.is_visible
        assert sm.state == TrackState.TRACK

        # Not visible in LOST
        sm.update(matched=False)
        assert not sm.is_visible
        assert sm.state == TrackState.LOST

        # Visible in RECOVER
        sm.update(matched=True)
        assert sm.is_visible
        assert sm.state == TrackState.RECOVER

    def test_force_drop(self):
        """force_drop should immediately transition to DROPPED."""
        sm = TrackStateMachine()
        sm.update(matched=True)  # LOCK

        reason = sm.force_drop("manual_stop")

        assert sm.state == TrackState.DROPPED
        assert reason == "manual_stop"
        assert not sm.is_active

    def test_consecutive_counters(self):
        """Should track consecutive matches/misses correctly within a state."""
        # Use a config that requires many matches to transition
        config = StateMachineConfig(lock_confirm_frames=10, lock_timeout_frames=15)
        sm = TrackStateMachine(config)

        # First match transitions SEARCH → LOCK, counters reset
        sm.update(matched=True)
        assert sm.state == TrackState.LOCK
        # Counters reset on state change
        assert sm.consecutive_matches == 0
        assert sm.consecutive_misses == 0

        # Now within LOCK state, track consecutive matches
        sm.update(matched=True)
        assert sm.consecutive_matches == 1
        assert sm.consecutive_misses == 0

        sm.update(matched=True)
        assert sm.consecutive_matches == 2

        # Miss resets match counter
        sm.update(matched=False)
        assert sm.consecutive_matches == 0
        assert sm.consecutive_misses == 1
