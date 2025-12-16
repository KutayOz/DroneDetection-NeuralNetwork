"""
Frame domain entities.

Contains business logic for video frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    """
    Domain entity for a video frame.

    Encapsulates image data with metadata and transformations.
    """

    frame_id: int
    image: np.ndarray  # HWC format
    timestamp_ms: int
    source_id: str = "default"
    pixel_format: str = "BGR"  # OpenCV default
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate frame."""
        if self.image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {self.image.ndim}D")

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        """Number of color channels."""
        if self.image.ndim == 2:
            return 1
        return self.image.shape[2]

    @property
    def shape(self) -> tuple:
        """Image shape (H, W, C)."""
        return self.image.shape

    @property
    def is_color(self) -> bool:
        """Whether frame is color (3 channels)."""
        return self.channels == 3

    @property
    def is_grayscale(self) -> bool:
        """Whether frame is grayscale."""
        return self.channels == 1

    def to_rgb(self) -> np.ndarray:
        """Convert to RGB format."""
        if self.pixel_format == "RGB":
            return self.image.copy()
        elif self.pixel_format == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif self.pixel_format == "GRAY":
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unknown pixel format: {self.pixel_format}")

    def to_bgr(self) -> np.ndarray:
        """Convert to BGR format."""
        if self.pixel_format == "BGR":
            return self.image.copy()
        elif self.pixel_format == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif self.pixel_format == "GRAY":
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unknown pixel format: {self.pixel_format}")

    def to_grayscale(self) -> np.ndarray:
        """Convert to grayscale."""
        if self.is_grayscale:
            return self.image.copy()
        if self.pixel_format == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def resize(self, width: int, height: int) -> Frame:
        """Return resized frame."""
        resized = cv2.resize(self.image, (width, height))
        return Frame(
            frame_id=self.frame_id,
            image=resized,
            timestamp_ms=self.timestamp_ms,
            source_id=self.source_id,
            pixel_format=self.pixel_format,
            metadata=self.metadata.copy(),
        )

    def crop(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop region from frame."""
        x1 = max(0, min(x1, self.width))
        y1 = max(0, min(y1, self.height))
        x2 = max(0, min(x2, self.width))
        y2 = max(0, min(y2, self.height))
        return self.image[y1:y2, x1:x2].copy()

    @classmethod
    def from_file(cls, path: str, frame_id: int = 0) -> Frame:
        """Load frame from image file."""
        import time

        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return cls(
            frame_id=frame_id,
            image=image,
            timestamp_ms=int(time.time() * 1000),
            source_id=path,
            pixel_format="BGR",
        )


@dataclass(frozen=True)
class FrameMetadata:
    """
    Immutable frame metadata (without image data).

    Useful for logging and passing around without copying image.
    """

    frame_id: int
    timestamp_ms: int
    source_id: str
    width: int
    height: int
    pixel_format: str = "BGR"

    @classmethod
    def from_frame(cls, frame: Frame) -> FrameMetadata:
        """Extract metadata from Frame."""
        return cls(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            source_id=frame.source_id,
            width=frame.width,
            height=frame.height,
            pixel_format=frame.pixel_format,
        )

    def compute_fps_from(self, other: FrameMetadata) -> float:
        """Compute FPS from another frame's metadata."""
        frame_diff = self.frame_id - other.frame_id
        time_diff_ms = self.timestamp_ms - other.timestamp_ms

        if time_diff_ms <= 0:
            return 0.0

        return (frame_diff * 1000) / time_diff_ms
