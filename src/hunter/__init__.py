"""
Hunter Drone Detection & Tracking System.

A real-time drone detection and tracking system using:
- YOLO11 for object detection
- Siamese networks for appearance embedding
- Kalman filtering for motion prediction
- Hungarian algorithm for track association

Example usage:
    from hunter import Pipeline, HunterConfig

    config = HunterConfig.from_yaml("config.yaml")

    with Pipeline(config) as pipeline:
        for message in pipeline.run():
            print(f"Frame {message.frame_id}: {message.track_count} tracks")
"""

__version__ = "1.0.0"
__author__ = "Hunter Drone Team"

# Core configuration
from .core import (
    HunterConfig,
    IngestConfig,
    PreprocessConfig,
    DetectorConfig,
    EmbedderConfig,
    TrackingConfig,
    OutputConfig,
    LoggingConfig,
)

# Exceptions
from .core import (
    HunterError,
    ConfigError,
    IngestError,
    ModelError,
    TrackingError,
)

# Main pipeline
from .pipeline import Pipeline, TrackMessage, TrackInfo

# Tracking
from .tracking import MultiTargetTracker, Track, TrackState

# Models
from .models import YOLODetector, SiameseEmbedder

__all__ = [
    # Version
    "__version__",
    # Config
    "HunterConfig",
    "IngestConfig",
    "PreprocessConfig",
    "DetectorConfig",
    "EmbedderConfig",
    "TrackingConfig",
    "OutputConfig",
    "LoggingConfig",
    # Exceptions
    "HunterError",
    "ConfigError",
    "IngestError",
    "ModelError",
    "TrackingError",
    # Pipeline
    "Pipeline",
    "TrackMessage",
    "TrackInfo",
    # Tracking
    "MultiTargetTracker",
    "Track",
    "TrackState",
    # Models
    "YOLODetector",
    "SiameseEmbedder",
]
