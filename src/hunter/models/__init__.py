"""
Deep learning models.

Provides:
- YOLO11 detector for object detection
- Siamese embedder for appearance features
"""

from .detector import BaseDetector, DetectionOutput, StubDetector, YOLODetector
from .embedder import BaseEmbedder, SiameseEmbedder, StubEmbedder

__all__ = [
    # Detector
    "BaseDetector",
    "DetectionOutput",
    "YOLODetector",
    "StubDetector",
    # Embedder
    "BaseEmbedder",
    "SiameseEmbedder",
    "StubEmbedder",
]
