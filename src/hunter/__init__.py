"""
Hunter Drone Detection & Tracking System.

A real-time drone detection and tracking system using:
- YOLO11 primary detector with hybrid routing
- Siamese network for secondary verification and re-identification
- Kalman filtering (filterpy) for motion prediction
- Hungarian algorithm for track association

Quick Start:
    from hunter import Pipeline, HunterConfig

    config = HunterConfig.from_yaml("configs/default.yaml")
    config.ingest.source_uri = "video.mp4"

    with Pipeline(config) as pipeline:
        for message in pipeline.run():
            print(f"Frame {message.frame_id}: {message.track_count} tracks")

CLI Usage:
    hunter-run --config configs/default.yaml --source video.mp4

For more information:
    - Documentation: docs/user-guide.md
    - Configuration: docs/configuration.md
    - Training: docs/training.md
"""

__version__ = "1.0.0"
__author__ = "Hunter Drone Team"
__python_requires__ = ">=3.10,<3.11"

# =============================================================================
# Core Exports - Main classes users need
# =============================================================================

from hunter.core.config import (
    HunterConfig,
    IngestConfig,
    DetectorConfig,
    TrackingConfig,
    OutputConfig,
)
from hunter.core.exceptions import (
    HunterError,
    ConfigError,
    IngestError,
    ModelError,
    TrackingError,
)

# =============================================================================
# Pipeline Exports - For running detection
# =============================================================================

from hunter.pipeline import (
    Pipeline,
    TrackMessage,
    TrackInfo,
    TrajectoryPoint,
    ModelInfo,
)

# =============================================================================
# Training Advisor - For training analysis
# =============================================================================

from hunter.training_advisor import (
    TrainingAdvisor,
    AdvisorConfig,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__python_requires__",
    # Core Configuration
    "HunterConfig",
    "IngestConfig",
    "DetectorConfig",
    "TrackingConfig",
    "OutputConfig",
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
    "TrajectoryPoint",
    "ModelInfo",
    # Training
    "TrainingAdvisor",
    "AdvisorConfig",
]


def get_version() -> str:
    """Get the Hunter Drone version string."""
    return __version__


def print_info() -> None:
    """Print Hunter Drone system information."""
    import sys

    print(f"Hunter Drone Detection & Tracking System v{__version__}")
    print(f"Python: {sys.version}")
    print(f"Required Python: {__python_requires__}")
    print()
    print("Quick Start:")
    print("  hunter-run --config configs/default.yaml --source video.mp4")
    print()
    print("Documentation:")
    print("  - User Guide: docs/user-guide.md")
    print("  - Configuration: docs/configuration.md")
    print("  - Training: docs/training.md")
