"""
Contract tests for tracker interfaces.

These tests verify that implementations correctly fulfill
their Protocol contracts for tracking components.
"""

import numpy as np
import pytest
from typing import List, Optional, Tuple

from hunter.interfaces.detector import BoundingBox, DetectionResult
from hunter.interfaces.tracker import (
    IMotionModel,
    IStateMachine,
    IAssociator,
    ITracker,
    TrackState,
    TrackSnapshot,
    TrajectoryPoint,
    AssociationResult,
)


# ============================================
# Mock implementations for contract testing
# ============================================


class MockMotionModel:
    """Mock motion model that fulfills IMotionModel contract."""

    def __init__(self):
        self._state = np.zeros(8, dtype=np.float32)  # [cx, cy, w, h, vx, vy, vw, vh]
        self._initialized = False

    def initialize(self, bbox: BoundingBox) -> None:
        cx, cy = bbox.center
        self._state = np.array(
            [cx, cy, bbox.width, bbox.height, 0, 0, 0, 0], dtype=np.float32
        )
        self._initialized = True

    def predict(self) -> BoundingBox:
        if not self._initialized:
            raise RuntimeError("Motion model not initialized")
        # Simple constant velocity prediction
        self._state[:4] += self._state[4:]
        cx, cy, w, h = self._state[:4]
        return BoundingBox.from_cxcywh(cx, cy, w, h)

    def update(self, bbox: BoundingBox) -> BoundingBox:
        if not self._initialized:
            self.initialize(bbox)
            return bbox
        cx, cy = bbox.center
        # Simple update (no actual Kalman)
        self._state[:4] = [cx, cy, bbox.width, bbox.height]
        return bbox

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._state[4]), float(self._state[5]))

    @property
    def state_vector(self) -> np.ndarray:
        return self._state.copy()


class MockStateMachine:
    """Mock state machine that fulfills IStateMachine contract."""

    def __init__(
        self,
        confirm_frames: int = 3,
        lost_frames: int = 30,
    ):
        self._state = TrackState.TENTATIVE
        self._confirm_frames = confirm_frames
        self._lost_frames = lost_frames
        self._hits = 0
        self._age = 0
        self._time_since_update = 0
        self._consecutive_hits = 0

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.is_active()

    @property
    def is_visible(self) -> bool:
        return self._state.is_visible()

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def age(self) -> int:
        return self._age

    @property
    def time_since_update(self) -> int:
        return self._time_since_update

    def update(self, matched: bool) -> Tuple[TrackState, Optional[str]]:
        old_state = self._state
        self._age += 1
        reason = None

        if matched:
            self._hits += 1
            self._consecutive_hits += 1
            self._time_since_update = 0

            if self._state == TrackState.TENTATIVE:
                if self._consecutive_hits >= self._confirm_frames:
                    self._state = TrackState.CONFIRMED
                    reason = "confirmed"
            elif self._state == TrackState.LOST:
                self._state = TrackState.CONFIRMED
                reason = "recovered"
        else:
            self._consecutive_hits = 0
            self._time_since_update += 1

            if self._state == TrackState.CONFIRMED:
                if self._time_since_update >= self._lost_frames:
                    self._state = TrackState.LOST
                    reason = "lost"
            elif self._state == TrackState.LOST:
                if self._time_since_update >= self._lost_frames:
                    self._state = TrackState.DELETED
                    reason = "deleted"

        if old_state != self._state:
            return self._state, reason
        return self._state, None

    def force_delete(self, reason: str) -> None:
        self._state = TrackState.DELETED


class MockAssociator:
    """Mock associator that fulfills IAssociator contract."""

    def __init__(self, iou_threshold: float = 0.3):
        self._iou_threshold = iou_threshold

    def associate(
        self,
        tracks: List[TrackSnapshot],
        detections: List[DetectionResult],
    ) -> AssociationResult:
        # Simple greedy association for testing
        matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))

        for t_idx, track in enumerate(tracks):
            best_d_idx = None
            best_iou = self._iou_threshold

            for d_idx in unmatched_detections:
                det = detections[d_idx]
                iou = self._compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx

            if best_d_idx is not None:
                matched_pairs.append((t_idx, best_d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(best_d_idx)

        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
        )

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0


class MockTracker:
    """Mock tracker that fulfills ITracker contract."""

    def __init__(self):
        self._tracks: List[TrackSnapshot] = []
        self._next_id = 0

    def update(
        self,
        detections: List[DetectionResult],
        timestamp_ms: int,
    ) -> List[TrackSnapshot]:
        # Create new tracks for detections
        new_tracks = []
        for det in detections:
            track = TrackSnapshot(
                track_id=self._next_id,
                state=TrackState.CONFIRMED,
                bbox=det.bbox,
                predicted_bbox=det.bbox,
                confidence=det.confidence,
                velocity=(0.0, 0.0),
                age_frames=1,
                hits=1,
                time_since_update=0,
            )
            new_tracks.append(track)
            self._next_id += 1

        self._tracks = new_tracks
        return self._tracks

    def get_visible_tracks(self) -> List[TrackSnapshot]:
        return [t for t in self._tracks if t.state.is_visible()]

    def get_all_tracks(self) -> List[TrackSnapshot]:
        return self._tracks

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    @property
    def visible_track_count(self) -> int:
        return len(self.get_visible_tracks())

    def get_stats(self) -> dict:
        return {
            "total_tracks": self._next_id,
            "active_tracks": self.track_count,
            "visible_tracks": self.visible_track_count,
        }


# ============================================
# Contract Tests
# ============================================


class TestTrackState:
    """Tests for TrackState enumeration."""

    def test_all_states_exist(self):
        """TrackState should have all required states."""
        assert hasattr(TrackState, "TENTATIVE")
        assert hasattr(TrackState, "CONFIRMED")
        assert hasattr(TrackState, "LOST")
        assert hasattr(TrackState, "DELETED")

    def test_is_active(self):
        """is_active should return True for active states."""
        assert TrackState.TENTATIVE.is_active()
        assert TrackState.CONFIRMED.is_active()
        assert TrackState.LOST.is_active()
        assert not TrackState.DELETED.is_active()

    def test_is_visible(self):
        """is_visible should return True for visible states."""
        assert not TrackState.TENTATIVE.is_visible()
        assert TrackState.CONFIRMED.is_visible()
        assert TrackState.LOST.is_visible()
        assert not TrackState.DELETED.is_visible()


class TestIMotionModelContract:
    """Contract tests for IMotionModel protocol."""

    @pytest.fixture
    def motion_model(self) -> MockMotionModel:
        return MockMotionModel()

    @pytest.fixture
    def sample_bbox(self) -> BoundingBox:
        return BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)

    def test_initialize(self, motion_model: MockMotionModel, sample_bbox: BoundingBox):
        """IMotionModel.initialize should initialize state."""
        motion_model.initialize(sample_bbox)
        # Should not raise
        assert motion_model.state_vector is not None

    def test_predict_returns_bbox(
        self, motion_model: MockMotionModel, sample_bbox: BoundingBox
    ):
        """IMotionModel.predict should return BoundingBox."""
        motion_model.initialize(sample_bbox)
        result = motion_model.predict()
        assert isinstance(result, BoundingBox)

    def test_update_returns_bbox(
        self, motion_model: MockMotionModel, sample_bbox: BoundingBox
    ):
        """IMotionModel.update should return BoundingBox."""
        motion_model.initialize(sample_bbox)
        result = motion_model.update(sample_bbox)
        assert isinstance(result, BoundingBox)

    def test_velocity_property(
        self, motion_model: MockMotionModel, sample_bbox: BoundingBox
    ):
        """IMotionModel.velocity should return (vx, vy) tuple."""
        motion_model.initialize(sample_bbox)
        vx, vy = motion_model.velocity
        assert isinstance(vx, float)
        assert isinstance(vy, float)

    def test_state_vector_property(
        self, motion_model: MockMotionModel, sample_bbox: BoundingBox
    ):
        """IMotionModel.state_vector should return numpy array."""
        motion_model.initialize(sample_bbox)
        state = motion_model.state_vector
        assert isinstance(state, np.ndarray)
        assert len(state) == 8  # [cx, cy, w, h, vx, vy, vw, vh]


class TestIStateMachineContract:
    """Contract tests for IStateMachine protocol."""

    @pytest.fixture
    def state_machine(self) -> MockStateMachine:
        return MockStateMachine(confirm_frames=3, lost_frames=30)

    def test_initial_state_is_tentative(self, state_machine: MockStateMachine):
        """IStateMachine should start in TENTATIVE state."""
        assert state_machine.state == TrackState.TENTATIVE

    def test_state_property(self, state_machine: MockStateMachine):
        """IStateMachine.state should return TrackState."""
        assert isinstance(state_machine.state, TrackState)

    def test_is_active_property(self, state_machine: MockStateMachine):
        """IStateMachine.is_active should return bool."""
        assert isinstance(state_machine.is_active, bool)

    def test_is_visible_property(self, state_machine: MockStateMachine):
        """IStateMachine.is_visible should return bool."""
        assert isinstance(state_machine.is_visible, bool)

    def test_hits_property(self, state_machine: MockStateMachine):
        """IStateMachine.hits should return int."""
        assert isinstance(state_machine.hits, int)
        assert state_machine.hits >= 0

    def test_age_property(self, state_machine: MockStateMachine):
        """IStateMachine.age should return int."""
        assert isinstance(state_machine.age, int)
        assert state_machine.age >= 0

    def test_time_since_update_property(self, state_machine: MockStateMachine):
        """IStateMachine.time_since_update should return int."""
        assert isinstance(state_machine.time_since_update, int)
        assert state_machine.time_since_update >= 0

    def test_update_returns_tuple(self, state_machine: MockStateMachine):
        """IStateMachine.update should return (state, reason) tuple."""
        result = state_machine.update(matched=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TrackState)
        assert result[1] is None or isinstance(result[1], str)

    def test_update_with_match_increases_hits(self, state_machine: MockStateMachine):
        """IStateMachine.update(matched=True) should increase hits."""
        initial_hits = state_machine.hits
        state_machine.update(matched=True)
        assert state_machine.hits == initial_hits + 1

    def test_transition_tentative_to_confirmed(self, state_machine: MockStateMachine):
        """Enough consecutive hits should transition to CONFIRMED."""
        for _ in range(3):
            state_machine.update(matched=True)
        assert state_machine.state == TrackState.CONFIRMED

    def test_force_delete(self, state_machine: MockStateMachine):
        """IStateMachine.force_delete should set state to DELETED."""
        state_machine.force_delete("test reason")
        assert state_machine.state == TrackState.DELETED


class TestIAssociatorContract:
    """Contract tests for IAssociator protocol."""

    @pytest.fixture
    def associator(self) -> MockAssociator:
        return MockAssociator(iou_threshold=0.3)

    @pytest.fixture
    def sample_tracks(self) -> List[TrackSnapshot]:
        return [
            TrackSnapshot(
                track_id=0,
                state=TrackState.CONFIRMED,
                bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                predicted_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                confidence=0.9,
                velocity=(0.0, 0.0),
                age_frames=10,
                hits=10,
                time_since_update=0,
            )
        ]

    @pytest.fixture
    def sample_detections(self) -> List[DetectionResult]:
        return [
            DetectionResult(
                bbox=BoundingBox(110.0, 110.0, 210.0, 210.0),
                confidence=0.9,
                class_id=0,
            )
        ]

    def test_associate_returns_result(
        self,
        associator: MockAssociator,
        sample_tracks: List[TrackSnapshot],
        sample_detections: List[DetectionResult],
    ):
        """IAssociator.associate should return AssociationResult."""
        result = associator.associate(sample_tracks, sample_detections)
        assert isinstance(result, AssociationResult)

    def test_association_result_has_required_fields(
        self,
        associator: MockAssociator,
        sample_tracks: List[TrackSnapshot],
        sample_detections: List[DetectionResult],
    ):
        """AssociationResult should have all required fields."""
        result = associator.associate(sample_tracks, sample_detections)

        assert hasattr(result, "matched_pairs")
        assert hasattr(result, "unmatched_tracks")
        assert hasattr(result, "unmatched_detections")

        assert isinstance(result.matched_pairs, list)
        assert isinstance(result.unmatched_tracks, list)
        assert isinstance(result.unmatched_detections, list)

    def test_associate_with_matching_detection(
        self,
        associator: MockAssociator,
        sample_tracks: List[TrackSnapshot],
        sample_detections: List[DetectionResult],
    ):
        """Should match overlapping track and detection."""
        result = associator.associate(sample_tracks, sample_detections)

        # Should have one match
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0] == (0, 0)
        assert len(result.unmatched_tracks) == 0
        assert len(result.unmatched_detections) == 0

    def test_associate_with_no_overlap(self, associator: MockAssociator):
        """Should not match non-overlapping track and detection."""
        tracks = [
            TrackSnapshot(
                track_id=0,
                state=TrackState.CONFIRMED,
                bbox=BoundingBox(0.0, 0.0, 50.0, 50.0),
                predicted_bbox=BoundingBox(0.0, 0.0, 50.0, 50.0),
                confidence=0.9,
                velocity=(0.0, 0.0),
                age_frames=10,
                hits=10,
                time_since_update=0,
            )
        ]
        detections = [
            DetectionResult(
                bbox=BoundingBox(500.0, 500.0, 600.0, 600.0),
                confidence=0.9,
                class_id=0,
            )
        ]

        result = associator.associate(tracks, detections)

        assert len(result.matched_pairs) == 0
        assert len(result.unmatched_tracks) == 1
        assert len(result.unmatched_detections) == 1


class TestITrackerContract:
    """Contract tests for ITracker protocol."""

    @pytest.fixture
    def tracker(self) -> MockTracker:
        return MockTracker()

    @pytest.fixture
    def sample_detections(self) -> List[DetectionResult]:
        return [
            DetectionResult(
                bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                confidence=0.9,
                class_id=0,
            )
        ]

    def test_update_returns_list_of_snapshots(
        self, tracker: MockTracker, sample_detections: List[DetectionResult]
    ):
        """ITracker.update should return List[TrackSnapshot]."""
        result = tracker.update(sample_detections, timestamp_ms=1000)

        assert isinstance(result, list)
        for snapshot in result:
            assert isinstance(snapshot, TrackSnapshot)

    def test_get_visible_tracks(
        self, tracker: MockTracker, sample_detections: List[DetectionResult]
    ):
        """ITracker.get_visible_tracks should return visible tracks."""
        tracker.update(sample_detections, timestamp_ms=1000)
        visible = tracker.get_visible_tracks()

        assert isinstance(visible, list)
        for track in visible:
            assert track.state.is_visible()

    def test_get_all_tracks(
        self, tracker: MockTracker, sample_detections: List[DetectionResult]
    ):
        """ITracker.get_all_tracks should return all tracks."""
        tracker.update(sample_detections, timestamp_ms=1000)
        all_tracks = tracker.get_all_tracks()

        assert isinstance(all_tracks, list)

    def test_reset_clears_tracks(
        self, tracker: MockTracker, sample_detections: List[DetectionResult]
    ):
        """ITracker.reset should clear all tracks."""
        tracker.update(sample_detections, timestamp_ms=1000)
        assert tracker.track_count > 0

        tracker.reset()
        assert tracker.track_count == 0

    def test_track_count_property(
        self, tracker: MockTracker, sample_detections: List[DetectionResult]
    ):
        """ITracker.track_count should return int."""
        assert isinstance(tracker.track_count, int)
        assert tracker.track_count == 0

        tracker.update(sample_detections, timestamp_ms=1000)
        assert tracker.track_count == len(sample_detections)

    def test_get_stats_returns_dict(self, tracker: MockTracker):
        """ITracker.get_stats should return dictionary."""
        stats = tracker.get_stats()
        assert isinstance(stats, dict)


class TestTrackSnapshot:
    """Tests for TrackSnapshot dataclass."""

    def test_creation(self):
        """TrackSnapshot can be created with required fields."""
        snapshot = TrackSnapshot(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
            predicted_bbox=BoundingBox(5.0, 5.0, 105.0, 105.0),
            confidence=0.9,
            velocity=(10.0, 5.0),
            age_frames=10,
            hits=8,
            time_since_update=2,
        )

        assert snapshot.track_id == 1
        assert snapshot.state == TrackState.CONFIRMED
        assert snapshot.confidence == 0.9

    def test_state_name_property(self):
        """TrackSnapshot.state_name should return string."""
        snapshot = TrackSnapshot(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
            predicted_bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
            confidence=0.9,
            velocity=(0.0, 0.0),
            age_frames=1,
            hits=1,
            time_since_update=0,
        )

        assert snapshot.state_name == "CONFIRMED"

    def test_optional_embedding(self):
        """TrackSnapshot can include optional embedding."""
        embedding = np.zeros(128, dtype=np.float32)
        snapshot = TrackSnapshot(
            track_id=1,
            state=TrackState.CONFIRMED,
            bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
            predicted_bbox=BoundingBox(0.0, 0.0, 100.0, 100.0),
            confidence=0.9,
            velocity=(0.0, 0.0),
            age_frames=1,
            hits=1,
            time_since_update=0,
            embedding=embedding,
        )

        assert snapshot.embedding is not None


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint dataclass."""

    def test_creation(self):
        """TrajectoryPoint can be created."""
        point = TrajectoryPoint(
            timestamp_ms=1000,
            cx=150.0,
            cy=200.0,
            confidence=0.95,
        )

        assert point.timestamp_ms == 1000
        assert point.cx == 150.0
        assert point.cy == 200.0
        assert point.confidence == 0.95

    def test_default_confidence(self):
        """TrajectoryPoint has default confidence of 1.0."""
        point = TrajectoryPoint(timestamp_ms=1000, cx=100.0, cy=100.0)
        assert point.confidence == 1.0
