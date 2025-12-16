"""
Pipeline modules.

Provides the main orchestration pipeline and all components.
"""

from .ingest import BaseIngest, FileIngest, FramePacket, StubIngest
from .output import (
    BaseOutput,
    CallbackSink,
    JsonFileSink,
    ModelInfo,
    StubSink,
    TrackInfo,
    TrackMessage,
    TrajectoryPoint,
)
from .orchestrator import Pipeline
from .preprocess import Preprocessor, crop_bbox, letterbox

__all__ = [
    # Main orchestrator
    "Pipeline",
    # Ingest
    "BaseIngest",
    "FileIngest",
    "StubIngest",
    "FramePacket",
    # Preprocess
    "Preprocessor",
    "letterbox",
    "crop_bbox",
    # Output
    "BaseOutput",
    "StubSink",
    "JsonFileSink",
    "CallbackSink",
    "TrackMessage",
    "TrackInfo",
    "ModelInfo",
    "TrajectoryPoint",
]
