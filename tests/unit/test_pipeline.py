"""
Unit tests for pipeline module.

Tests for ingest, output sinks, and pipeline orchestration.
"""

import time
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hunter.pipeline.ingest.base import BaseIngest
from hunter.pipeline.ingest.frame_packet import FramePacket
from hunter.pipeline.ingest.file_ingest import StubIngest
from hunter.pipeline.output.base import BaseOutput
from hunter.pipeline.output.stub_sink import StubSink
from hunter.pipeline.output.track_message import (
    TrackMessage,
    TrackInfo,
    ModelInfo,
    TrajectoryPoint,
)


# ============================================
# FramePacket Tests
# ============================================


class TestFramePacket:
    """Tests for FramePacket data class."""

    def test_creation(self):
        """FramePacket can be created."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            frame_id=0,
            image=image,
            width=640,
            height=480,
            recv_ts_ms=int(time.time() * 1000),
            decode_ts_ms=int(time.time() * 1000),
            pixel_format="BGR",
            source_id="test",
        )
        assert packet.frame_id == 0
        assert packet.width == 640
        assert packet.height == 480

    def test_effective_timestamp(self):
        """Effective timestamp returns correct value."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        recv_ts = int(time.time() * 1000)
        packet = FramePacket(
            frame_id=0,
            image=image,
            width=640,
            height=480,
            recv_ts_ms=recv_ts,
            decode_ts_ms=recv_ts + 10,
            pixel_format="BGR",
            source_id="test",
        )
        # Should return either recv or decode timestamp
        assert packet.effective_timestamp >= recv_ts


# ============================================
# StubIngest Tests
# ============================================


class TestStubIngest:
    """Tests for StubIngest."""

    def test_creates_frames(self):
        """StubIngest generates frames."""
        ingest = StubIngest(num_frames=5)
        frames = list(ingest)
        assert len(frames) == 5

    def test_frame_properties(self):
        """StubIngest frames have correct properties."""
        ingest = StubIngest(num_frames=1, width=640, height=480, fps=30.0)
        frame = next(iter(ingest))

        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame_id == 0

    def test_fps_property(self):
        """StubIngest returns correct FPS."""
        ingest = StubIngest(fps=25.0)
        assert ingest.fps == 25.0

    def test_frame_count_property(self):
        """StubIngest returns correct frame count."""
        ingest = StubIngest(num_frames=100)
        assert ingest.frame_count == 100

    def test_accepts_custom_frames(self):
        """StubIngest accepts custom frame list."""
        custom_frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
        ]
        ingest = StubIngest(frames=custom_frames)

        assert ingest.frame_count == 2
        frames = list(ingest)
        assert len(frames) == 2

    def test_iterator_protocol(self):
        """StubIngest implements iterator protocol."""
        ingest = StubIngest(num_frames=3)

        assert hasattr(ingest, '__iter__')
        assert hasattr(ingest, '__next__')

        # Can iterate
        count = 0
        for frame in ingest:
            count += 1
        assert count == 3

    def test_close(self):
        """StubIngest close stops iteration."""
        ingest = StubIngest(num_frames=10)
        ingest.close()

        with pytest.raises(StopIteration):
            next(ingest)


# ============================================
# StubSink Tests
# ============================================


class TestStubSink:
    """Tests for StubSink output."""

    def test_write_message(self):
        """StubSink accepts messages."""
        from hunter.core.config import OutputConfig
        config = OutputConfig(sink_type="stub")
        sink = StubSink(config)

        message = self._create_message()
        sink.write(message)

        assert sink.message_count == 1

    def test_close(self):
        """StubSink close works."""
        from hunter.core.config import OutputConfig
        config = OutputConfig(sink_type="stub")
        sink = StubSink(config)

        sink.close()  # Should not raise

    def test_messages_property(self):
        """StubSink stores messages for retrieval."""
        from hunter.core.config import OutputConfig
        config = OutputConfig(sink_type="stub")
        sink = StubSink(config)

        message1 = self._create_message(frame_id=0)
        message2 = self._create_message(frame_id=1)

        sink.write(message1)
        sink.write(message2)

        messages = sink.messages
        assert len(messages) == 2

    def _create_message(self, frame_id: int = 0) -> TrackMessage:
        """Create test TrackMessage."""
        model_info = ModelInfo(
            detector_name="TestDetector",
            detector_hash="abc123",
        )
        return TrackMessage.create(
            frame_id=frame_id,
            model=model_info,
            pipeline_metrics={},
            tracks=[],
        )


# ============================================
# TrackMessage Tests
# ============================================


class TestTrackMessage:
    """Tests for TrackMessage."""

    def test_create(self):
        """TrackMessage.create() works."""
        model_info = ModelInfo(
            detector_name="TestDetector",
            detector_hash="abc123",
        )
        message = TrackMessage.create(
            frame_id=0,
            model=model_info,
            pipeline_metrics={"detect_ms": 10.0},
            tracks=[],
        )

        assert message.frame_id == 0
        assert message.track_count == 0

    def test_with_tracks(self):
        """TrackMessage with tracks."""
        model_info = ModelInfo(
            detector_name="TestDetector",
            detector_hash="abc123",
        )
        track_info = TrackInfo(
            track_id=1,
            state="CONFIRMED",
            confidence=0.9,
            bbox_xyxy=(100, 100, 200, 200),
            predicted_bbox_xyxy=(105, 105, 205, 205),
            velocity_px_per_s=(10.0, 5.0),
            trajectory_tail=[],
            age_frames=10,
            hits=8,
            time_since_update=0,
        )
        message = TrackMessage.create(
            frame_id=0,
            model=model_info,
            pipeline_metrics={},
            tracks=[track_info],
        )

        assert message.track_count == 1

    def test_to_dict(self):
        """TrackMessage serializes to dict."""
        model_info = ModelInfo(
            detector_name="TestDetector",
            detector_hash="abc123",
        )
        message = TrackMessage.create(
            frame_id=0,
            model=model_info,
            pipeline_metrics={},
            tracks=[],
        )

        d = message.to_dict()
        assert isinstance(d, dict)
        assert "frame_id" in d


class TestTrackInfo:
    """Tests for TrackInfo."""

    def test_creation(self):
        """TrackInfo can be created."""
        info = TrackInfo(
            track_id=1,
            state="CONFIRMED",
            confidence=0.9,
            bbox_xyxy=(100, 100, 200, 200),
            predicted_bbox_xyxy=(105, 105, 205, 205),
            velocity_px_per_s=(10.0, 5.0),
            trajectory_tail=[],
            age_frames=10,
            hits=8,
            time_since_update=0,
        )
        assert info.track_id == 1
        assert info.state == "CONFIRMED"

    def test_with_trajectory(self):
        """TrackInfo with trajectory points."""
        trajectory = [
            TrajectoryPoint(t_ms=1000, cx=100.0, cy=100.0),
            TrajectoryPoint(t_ms=2000, cx=110.0, cy=105.0),
        ]
        info = TrackInfo(
            track_id=1,
            state="CONFIRMED",
            confidence=0.9,
            bbox_xyxy=(100, 100, 200, 200),
            predicted_bbox_xyxy=(105, 105, 205, 205),
            velocity_px_per_s=(10.0, 5.0),
            trajectory_tail=trajectory,
            age_frames=10,
            hits=8,
            time_since_update=0,
        )
        assert len(info.trajectory_tail) == 2


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint."""

    def test_creation(self):
        """TrajectoryPoint can be created."""
        point = TrajectoryPoint(t_ms=1000, cx=100.0, cy=200.0)
        assert point.t_ms == 1000
        assert point.cx == 100.0
        assert point.cy == 200.0


class TestModelInfo:
    """Tests for ModelInfo."""

    def test_creation(self):
        """ModelInfo can be created."""
        info = ModelInfo(
            detector_name="YOLO11",
            detector_hash="abc123",
        )
        assert info.detector_name == "YOLO11"
        assert info.detector_hash == "abc123"

    def test_with_embedder(self):
        """ModelInfo with embedder info."""
        info = ModelInfo(
            detector_name="YOLO11",
            detector_hash="abc123",
            embedder_name="Siamese",
            embedder_hash="def456",
        )
        assert info.embedder_name == "Siamese"
        assert info.embedder_hash == "def456"


# ============================================
# BaseIngest Protocol Tests
# ============================================


class TestBaseIngestProtocol:
    """Tests for BaseIngest interface compliance."""

    def test_stub_inherits_base(self):
        """StubIngest inherits from BaseIngest."""
        ingest = StubIngest(num_frames=1)
        assert isinstance(ingest, BaseIngest)

    def test_has_required_methods(self):
        """BaseIngest implementations have required methods."""
        ingest = StubIngest(num_frames=1)

        # Required properties
        assert hasattr(ingest, 'fps')
        assert hasattr(ingest, 'frame_count')
        assert hasattr(ingest, 'is_live')

        # Required methods
        assert hasattr(ingest, '__iter__')
        assert hasattr(ingest, '__next__')
        assert hasattr(ingest, 'close')

    def test_context_manager(self):
        """BaseIngest supports context manager."""
        with StubIngest(num_frames=5) as ingest:
            frames = list(ingest)
            assert len(frames) == 5


# ============================================
# BaseOutput Protocol Tests
# ============================================


class TestBaseOutputProtocol:
    """Tests for BaseOutput interface compliance."""

    def test_stub_inherits_base(self):
        """StubSink inherits from BaseOutput."""
        from hunter.core.config import OutputConfig
        config = OutputConfig(sink_type="stub")
        sink = StubSink(config)
        assert isinstance(sink, BaseOutput)

    def test_has_required_methods(self):
        """BaseOutput implementations have required methods."""
        from hunter.core.config import OutputConfig
        config = OutputConfig(sink_type="stub")
        sink = StubSink(config)

        # Required methods
        assert hasattr(sink, 'write')
        assert hasattr(sink, 'close')
