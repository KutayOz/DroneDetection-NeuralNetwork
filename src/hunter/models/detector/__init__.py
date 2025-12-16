"""
Object detection models.

Provides YOLO11-based detector and abstract interface.
"""

from .base import BaseDetector, DetectionOutput
from .yolo_detector import StubDetector, YOLODetector

__all__ = [
    "BaseDetector",
    "DetectionOutput",
    "YOLODetector",
    "StubDetector",
]
