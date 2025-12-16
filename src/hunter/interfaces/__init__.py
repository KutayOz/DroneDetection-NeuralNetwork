"""
Interface definitions (Protocol classes) for the Hunter Drone system.

This module contains abstract interfaces following the Dependency Inversion Principle.
All concrete implementations depend on these interfaces, not each other.

Interface Categories:
- Detection: IDetector, IEmbedder, IVerifier, IDetectionRouter
- Tracking: ITracker, IAssociator, IStateMachine, IMotionModel
- Pipeline: IIngest, IPreprocessor, IOutputSink
"""

# Detection interfaces
from .detector import (
    BoundingBox,
    DetectionResult,
    VerificationResult,
    IDetector,
    IEmbedder,
    IVerifier,
    IDetectionRouter,
)

# Tracking interfaces
from .tracker import (
    ITracker,
    IAssociator,
    IStateMachine,
    IMotionModel,
)

# Pipeline interfaces
from .pipeline import (
    IIngest,
    IPreprocessor,
    IOutputSink,
)

__all__ = [
    # Value objects
    "BoundingBox",
    "DetectionResult",
    "VerificationResult",
    # Detection interfaces
    "IDetector",
    "IEmbedder",
    "IVerifier",
    "IDetectionRouter",
    # Tracking interfaces
    "ITracker",
    "IAssociator",
    "IStateMachine",
    "IMotionModel",
    # Pipeline interfaces
    "IIngest",
    "IPreprocessor",
    "IOutputSink",
]
