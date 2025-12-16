"""
Pytest configuration and fixtures.

Provides common fixtures for all tests.
"""

import pytest
import numpy as np
from pathlib import Path

from hunter.core.config import (
    HunterConfig,
    IngestConfig,
    PreprocessConfig,
    DetectorConfig,
    EmbedderConfig,
    TrackingConfig,
    OutputConfig,
    LoggingConfig,
)
from hunter.tracking import (
    Track,
    TrackConfig,
    StateMachineConfig,
    MultiTargetTracker,
    Detection,
)
from hunter.tracking.kalman_filter import KalmanConfig
from hunter.pipeline.ingest.frame_packet import FramePacket


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return (100.0, 100.0, 200.0, 200.0)


@pytest.fixture
def sample_frame_packet(sample_image):
    """Create a sample FramePacket."""
    return FramePacket.from_numpy(
        frame_id=0,
        image=sample_image,
        source_id="test",
    )


@pytest.fixture
def tracking_config():
    """Create a test tracking config."""
    return TrackingConfig(
        lock_confirm_frames=2,
        lock_timeout_frames=3,
        lost_timeout_frames=10,
        recover_max_frames=5,
        recover_confirm_frames=2,
        process_noise=1.0,
        measurement_noise=1.0,
        iou_threshold=0.3,
        embedding_weight=0.0,
        gate_threshold=0.1,
        trajectory_max_length=50,
        trajectory_output_points=5,
    )


@pytest.fixture
def track_config():
    """Create a test track config."""
    return TrackConfig(
        state_config=StateMachineConfig(
            lock_confirm_frames=2,
            lock_timeout_frames=3,
            lost_timeout_frames=10,
            recover_max_frames=5,
            recover_confirm_frames=2,
        ),
        kalman_config=KalmanConfig(
            process_noise=1.0,
            measurement_noise=1.0,
            dt=1/30,
        ),
        trajectory_max_length=50,
        embedding_ema_alpha=0.3,
    )


@pytest.fixture
def sample_detection(sample_bbox):
    """Create a sample detection."""
    return Detection(
        bbox_xyxy=sample_bbox,
        confidence=0.9,
        class_id=0,
        embedding=None,
    )


@pytest.fixture
def multi_tracker(tracking_config):
    """Create a multi-target tracker."""
    tracker = MultiTargetTracker(tracking_config, fps=30.0)
    yield tracker
    tracker.reset()
    Track.reset_id_counter()


@pytest.fixture(autouse=True)
def reset_track_counter():
    """Reset track ID counter before each test."""
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create a temporary config file."""
    config_content = """
ingest:
  source_type: stub
  source_uri: ""
  buffer_size: 5
  timeout_ms: 5000

preprocess:
  input_size: [640, 640]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  pixel_format: RGB

detector:
  model_path: test.pt
  model_type: yolo
  confidence_threshold: 0.5
  nms_threshold: 0.45
  max_detections: 100
  device: cpu

embedder:
  enabled: false

tracking:
  lock_confirm_frames: 3
  lock_timeout_frames: 5
  lost_timeout_frames: 30
  recover_max_frames: 15
  process_noise: 1.0
  measurement_noise: 1.0
  iou_threshold: 0.3
  embedding_weight: 0.0
  gate_threshold: 0.1
  trajectory_max_length: 150
  trajectory_output_points: 10

output:
  sink_type: stub
  pretty_print: false

logging:
  level: WARNING
  format: console
  log_stage_timings: false
  log_track_changes: false
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file
