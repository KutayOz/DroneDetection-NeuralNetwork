"""
Video frame data structure.

Follows SRP: Only responsible for frame data encapsulation.
ICD 2.1 compliant.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FramePacket:
    """
    Decoded video frame with metadata.

    This is the primary data structure passed through the pipeline.
    Contains the image data and associated metadata.

    Attributes:
        frame_id: Unique frame identifier (monotonically increasing)
        image: Image data as numpy array (H, W, 3), dtype uint8
        width: Image width in pixels
        height: Image height in pixels
        recv_ts_ms: Timestamp when frame was received (milliseconds)
        decode_ts_ms: Timestamp when frame was decoded (milliseconds)
        pixel_format: Color format ("BGR" or "RGB")
        capture_ts_ms: Original capture timestamp (0 if not available)
        source_id: Identifier of the video source
    """

    frame_id: int
    image: np.ndarray
    width: int
    height: int
    recv_ts_ms: int
    decode_ts_ms: int
    pixel_format: str = "BGR"
    capture_ts_ms: int = 0
    source_id: str = "default"

    def __post_init__(self) -> None:
        """Validate frame packet data."""
        # Validate image shape
        if self.image.ndim != 3:
            raise ValueError(f"Image must be 3D, got shape: {self.image.shape}")

        if self.image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, got: {self.image.shape[2]}")

        # Validate dimensions match
        h, w = self.image.shape[:2]
        if h != self.height or w != self.width:
            raise ValueError(
                f"Image shape ({h}, {w}) doesn't match "
                f"dimensions ({self.height}, {self.width})"
            )

        # Validate pixel format
        if self.pixel_format not in ("BGR", "RGB"):
            raise ValueError(f"Invalid pixel format: {self.pixel_format}")

    @property
    def has_capture_timestamp(self) -> bool:
        """Check if original capture timestamp is available."""
        return self.capture_ts_ms > 0

    @property
    def effective_timestamp(self) -> int:
        """
        Get the most accurate timestamp for latency calculation.

        Uses capture timestamp if available, otherwise receive timestamp.
        """
        return self.capture_ts_ms if self.has_capture_timestamp else self.recv_ts_ms

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get image shape (H, W, C)."""
        return self.image.shape

    @property
    def aspect_ratio(self) -> float:
        """Get image aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 1.0

    @classmethod
    def from_numpy(
        cls,
        frame_id: int,
        image: np.ndarray,
        source_id: str = "default",
        pixel_format: str = "BGR",
        capture_ts_ms: int = 0,
    ) -> "FramePacket":
        """
        Factory method to create FramePacket from numpy array.

        Automatically fills in timestamps and dimensions.

        Args:
            frame_id: Frame identifier
            image: Image array (H, W, 3)
            source_id: Source identifier
            pixel_format: Color format
            capture_ts_ms: Original capture timestamp

        Returns:
            FramePacket instance
        """
        now_ms = int(time.time() * 1000)
        h, w = image.shape[:2]

        return cls(
            frame_id=frame_id,
            image=image,
            width=w,
            height=h,
            recv_ts_ms=now_ms,
            decode_ts_ms=now_ms,
            pixel_format=pixel_format,
            capture_ts_ms=capture_ts_ms,
            source_id=source_id,
        )

    @classmethod
    def create_dummy(
        cls,
        width: int = 640,
        height: int = 480,
        frame_id: int = 0,
    ) -> "FramePacket":
        """
        Create a dummy frame for testing.

        Args:
            width: Image width
            height: Image height
            frame_id: Frame ID

        Returns:
            FramePacket with random image data
        """
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return cls.from_numpy(frame_id, image)

    def to_dict(self) -> dict:
        """
        Serialize metadata (without image data).

        Useful for logging and debugging.
        """
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "capture_ts_ms": self.capture_ts_ms,
            "recv_ts_ms": self.recv_ts_ms,
            "decode_ts_ms": self.decode_ts_ms,
            "source_id": self.source_id,
        }

    def copy_with_image(self, new_image: np.ndarray) -> "FramePacket":
        """
        Create a copy with a different image.

        Useful for preprocessing that changes image size.

        Args:
            new_image: New image array

        Returns:
            New FramePacket with updated image and dimensions
        """
        h, w = new_image.shape[:2]
        return FramePacket(
            frame_id=self.frame_id,
            image=new_image,
            width=w,
            height=h,
            recv_ts_ms=self.recv_ts_ms,
            decode_ts_ms=self.decode_ts_ms,
            pixel_format=self.pixel_format,
            capture_ts_ms=self.capture_ts_ms,
            source_id=self.source_id,
        )

    def __repr__(self) -> str:
        return (
            f"FramePacket(id={self.frame_id}, "
            f"size={self.width}x{self.height}, "
            f"format={self.pixel_format})"
        )
