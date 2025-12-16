"""
Domain models and business logic.

This module contains the core domain entities and value objects
that represent the business domain of drone detection and tracking.

Domain Entities:
- Detection: A single frame detection
- VerifiedDetection: Detection with verification metadata
- DetectionBatch: Collection of detections from a frame
- Frame: Video frame with image data
- TrackIdentity: Immutable track identifier
- TrackMetrics: Track statistics

Value Objects:
- BoundingBox: Immutable bbox representation (from interfaces)
- TrackState: Track lifecycle state enum
"""

from .detection import Detection, VerifiedDetection, DetectionBatch
from .track import TrackState, TrackIdentity, TrackMetrics, TrackTransition
from .frame import Frame, FrameMetadata

# Re-export from interfaces for convenience
from ..interfaces.detector import BoundingBox

__all__ = [
    # Detection entities
    "Detection",
    "VerifiedDetection",
    "DetectionBatch",
    # Track entities
    "TrackState",
    "TrackIdentity",
    "TrackMetrics",
    "TrackTransition",
    # Frame entities
    "Frame",
    "FrameMetadata",
    # Value objects
    "BoundingBox",
]
