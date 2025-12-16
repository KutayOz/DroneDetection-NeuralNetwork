"""
Abstract detector interface.

Follows Interface Segregation Principle (ISP):
- Minimal interface that all detectors must implement
- Clients depend only on methods they use

Follows Liskov Substitution Principle (LSP):
- Any detector implementation can replace another
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DetectionOutput:
    """
    Single detection output from detector.

    Represents a detected object with bounding box and confidence.
    """

    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Default: drone class

    @property
    def center(self) -> Tuple[float, float]:
        """Get detection center point."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get detection width."""
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        """Get detection height."""
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> float:
        """Get detection area."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
        }


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    All detector implementations must inherit from this class
    and implement the abstract methods.

    This ensures all detectors have a consistent interface,
    following the Dependency Inversion Principle (DIP).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Model name (human-readable identifier).

        Returns:
            String identifier for the model
        """
        pass

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Model checksum (SHA256).

        Used for model versioning and integrity verification.

        Returns:
            Hexadecimal hash string
        """
        pass

    @property
    def version(self) -> str:
        """
        Short version identifier.

        Returns:
            First 8 characters of hash
        """
        return self.hash[:8] if self.hash else "unknown"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionOutput]:
        """
        Run object detection on image.

        Args:
            image: Input image. Can be:
                - BGR/RGB uint8 (H, W, 3)
                - Normalized float32 (H, W, 3)
                The specific format depends on implementation.

        Returns:
            List of DetectionOutput objects
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up the model.

        First inference is typically slow due to model loading
        and GPU memory allocation. Call this before timing-critical
        inference to ensure consistent latency.
        """
        pass

    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectionOutput]]:
        """
        Run detection on batch of images.

        Default implementation processes sequentially.
        Override for optimized batch processing.

        Args:
            images: List of input images

        Returns:
            List of detection lists (one per image)
        """
        return [self.detect(img) for img in images]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"
