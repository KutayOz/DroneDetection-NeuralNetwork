"""
Unit tests for domain entities.

Domain entities encapsulate business logic and invariants.
"""

import numpy as np
import pytest
from typing import List

from hunter.domain.detection import Detection, VerifiedDetection, DetectionBatch
from hunter.domain.track import TrackState, TrackIdentity, TrackMetrics
from hunter.domain.frame import Frame, FrameMetadata
from hunter.interfaces.detector import BoundingBox


class TestDetection:
    """Tests for Detection domain entity."""

    def test_creation(self):
        """Detection can be created with required fields."""
        det = Detection(
            bbox=BoundingBox(10, 20, 100, 150),
            confidence=0.9,
            class_id=0,
        )
        assert det.confidence == 0.9
        assert det.class_id == 0

    def test_with_embedding(self):
        """Detection can include embedding."""
        embedding = np.random.randn(128).astype(np.float32)
        det = Detection(
            bbox=BoundingBox(10, 20, 100, 150),
            confidence=0.9,
            class_id=0,
            embedding=embedding,
        )
        assert det.embedding is not None
        assert det.has_embedding

    def test_without_embedding(self):
        """Detection without embedding has has_embedding=False."""
        det = Detection(
            bbox=BoundingBox(10, 20, 100, 150),
            confidence=0.9,
            class_id=0,
        )
        assert not det.has_embedding

    def test_center_property(self):
        """Detection center matches bbox center."""
        det = Detection(
            bbox=BoundingBox(0, 0, 100, 100),
            confidence=0.9,
            class_id=0,
        )
        cx, cy = det.center
        assert cx == 50.0
        assert cy == 50.0

    def test_area_property(self):
        """Detection area matches bbox area."""
        det = Detection(
            bbox=BoundingBox(0, 0, 10, 20),
            confidence=0.9,
            class_id=0,
        )
        assert det.area == 200.0


class TestVerifiedDetection:
    """Tests for VerifiedDetection domain entity."""

    def test_creation(self):
        """VerifiedDetection includes verification metadata."""
        det = VerifiedDetection(
            bbox=BoundingBox(10, 20, 100, 150),
            confidence=0.9,
            class_id=0,
            is_verified=True,
            verification_score=0.85,
            routing_path="siamese",
        )
        assert det.is_verified
        assert det.verification_score == 0.85
        assert det.routing_path == "siamese"

    def test_direct_pass_routing(self):
        """Detection with direct routing has correct path."""
        det = VerifiedDetection(
            bbox=BoundingBox(10, 20, 100, 150),
            confidence=0.95,
            class_id=0,
            is_verified=True,
            verification_score=0.95,
            routing_path="direct",
        )
        assert det.routing_path == "direct"


class TestDetectionBatch:
    """Tests for DetectionBatch domain entity."""

    def test_creation(self):
        """DetectionBatch can hold multiple detections."""
        detections = [
            Detection(BoundingBox(0, 0, 50, 50), 0.9, 0),
            Detection(BoundingBox(100, 100, 150, 150), 0.8, 0),
        ]
        batch = DetectionBatch(
            frame_id=1,
            timestamp_ms=1000,
            detections=detections,
        )
        assert len(batch) == 2
        assert batch.frame_id == 1

    def test_empty_batch(self):
        """Empty DetectionBatch is valid."""
        batch = DetectionBatch(frame_id=1, timestamp_ms=1000, detections=[])
        assert len(batch) == 0
        assert batch.is_empty

    def test_iteration(self):
        """DetectionBatch supports iteration."""
        detections = [
            Detection(BoundingBox(0, 0, 50, 50), 0.9, 0),
            Detection(BoundingBox(100, 100, 150, 150), 0.8, 0),
        ]
        batch = DetectionBatch(frame_id=1, timestamp_ms=1000, detections=detections)

        collected = list(batch)
        assert len(collected) == 2

    def test_filter_by_confidence(self):
        """DetectionBatch can filter by confidence."""
        detections = [
            Detection(BoundingBox(0, 0, 50, 50), 0.9, 0),
            Detection(BoundingBox(100, 100, 150, 150), 0.3, 0),
            Detection(BoundingBox(200, 200, 250, 250), 0.7, 0),
        ]
        batch = DetectionBatch(frame_id=1, timestamp_ms=1000, detections=detections)

        filtered = batch.filter_by_confidence(0.5)
        assert len(filtered) == 2


class TestTrackState:
    """Tests for TrackState domain entity."""

    def test_state_names(self):
        """TrackState enum has expected values."""
        assert TrackState.TENTATIVE.value == 1
        assert TrackState.CONFIRMED.value == 2
        assert TrackState.LOST.value == 3
        assert TrackState.DELETED.value == 4


class TestTrackIdentity:
    """Tests for TrackIdentity domain entity."""

    def test_creation(self):
        """TrackIdentity assigns unique ID."""
        identity = TrackIdentity(track_id=1)
        assert identity.track_id == 1

    def test_equality(self):
        """TrackIdentity equality by ID."""
        id1 = TrackIdentity(track_id=1)
        id2 = TrackIdentity(track_id=1)
        id3 = TrackIdentity(track_id=2)

        assert id1 == id2
        assert id1 != id3

    def test_hash(self):
        """TrackIdentity is hashable."""
        id1 = TrackIdentity(track_id=1)
        id2 = TrackIdentity(track_id=1)

        # Same ID should have same hash
        assert hash(id1) == hash(id2)

        # Can use in sets
        s = {id1, id2}
        assert len(s) == 1


class TestTrackMetrics:
    """Tests for TrackMetrics domain entity."""

    def test_creation(self):
        """TrackMetrics tracks statistics."""
        metrics = TrackMetrics()
        assert metrics.age == 0
        assert metrics.hits == 0
        assert metrics.misses == 0

    def test_record_hit(self):
        """TrackMetrics records hits."""
        metrics = TrackMetrics()
        metrics.record_hit()
        metrics.record_hit()

        assert metrics.hits == 2
        assert metrics.age == 2

    def test_record_miss(self):
        """TrackMetrics records misses."""
        metrics = TrackMetrics()
        metrics.record_miss()

        assert metrics.misses == 1
        assert metrics.time_since_update == 1

    def test_hit_ratio(self):
        """TrackMetrics computes hit ratio."""
        metrics = TrackMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()

        assert metrics.hit_ratio == pytest.approx(2 / 3)

    def test_hit_ratio_zero_age(self):
        """TrackMetrics hit_ratio handles zero age."""
        metrics = TrackMetrics()
        assert metrics.hit_ratio == 0.0


class TestFrame:
    """Tests for Frame domain entity."""

    def test_creation(self):
        """Frame can be created with image data."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=0,
            image=image,
            timestamp_ms=1000,
        )
        assert frame.frame_id == 0
        assert frame.width == 640
        assert frame.height == 480

    def test_dimensions(self):
        """Frame provides correct dimensions."""
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(frame_id=0, image=image, timestamp_ms=1000)

        assert frame.width == 1280
        assert frame.height == 720
        assert frame.channels == 3

    def test_to_rgb(self):
        """Frame can convert to RGB."""
        # Create BGR image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Blue channel
        frame = Frame(frame_id=0, image=image, timestamp_ms=1000, pixel_format="BGR")

        rgb = frame.to_rgb()
        # After BGR->RGB, red channel should have the value
        assert rgb[:, :, 2].mean() == 255  # Original blue is now in red position


class TestFrameMetadata:
    """Tests for FrameMetadata domain entity."""

    def test_creation(self):
        """FrameMetadata holds frame metadata."""
        meta = FrameMetadata(
            frame_id=10,
            timestamp_ms=5000,
            source_id="camera_1",
            width=1920,
            height=1080,
        )
        assert meta.frame_id == 10
        assert meta.source_id == "camera_1"

    def test_fps_calculation(self):
        """FrameMetadata can compute FPS from timestamps."""
        meta1 = FrameMetadata(frame_id=0, timestamp_ms=0, source_id="test", width=640, height=480)
        meta2 = FrameMetadata(frame_id=30, timestamp_ms=1000, source_id="test", width=640, height=480)

        fps = meta2.compute_fps_from(meta1)
        assert fps == pytest.approx(30.0, rel=0.1)
