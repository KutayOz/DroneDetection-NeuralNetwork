"""
File-based video ingest.

Reads video files using OpenCV.
"""

import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import BaseIngest
from .frame_packet import FramePacket
from ...core.config import IngestConfig
from ...core.exceptions import IngestError


class FileIngest(BaseIngest):
    """
    Video file ingest using OpenCV.

    Supports common video formats:
    - MP4, AVI, MKV, MOV
    - Any format supported by FFmpeg

    Example usage:
        config = IngestConfig(source_type="file", source_uri="video.mp4")
        with FileIngest(config) as ingest:
            for frame in ingest:
                process(frame)
    """

    def __init__(self, config: IngestConfig):
        """
        Initialize file ingest.

        Args:
            config: Ingest configuration

        Raises:
            IngestError: If file not found or cannot be opened
        """
        self._config = config
        self._path = Path(config.source_uri)

        # Validate file exists
        if not self._path.exists():
            raise IngestError(f"Video file not found: {self._path}")

        # Open video capture
        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise IngestError(f"Failed to open video: {self._path}")

        # Get video properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # State
        self._frame_id = 0
        self._closed = False

    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._fps

    @property
    def frame_count(self) -> Optional[int]:
        """Total frame count."""
        return self._frame_count if self._frame_count > 0 else None

    @property
    def width(self) -> int:
        """Video width."""
        return self._width

    @property
    def height(self) -> int:
        """Video height."""
        return self._height

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        if self._frame_count > 0 and self._fps > 0:
            return self._frame_count / self._fps
        return 0.0

    @property
    def current_position(self) -> int:
        """Current frame position."""
        return self._frame_id

    @property
    def progress(self) -> float:
        """Progress as percentage (0-100)."""
        if self._frame_count > 0:
            return (self._frame_id / self._frame_count) * 100
        return 0.0

    def __iter__(self) -> Iterator[FramePacket]:
        """Get frame iterator."""
        return self

    def __next__(self) -> FramePacket:
        """
        Get next frame.

        Returns:
            FramePacket containing frame data

        Raises:
            StopIteration: When video ends
            IngestError: On read error
        """
        if self._closed:
            raise StopIteration

        recv_ts = int(time.time() * 1000)

        # Read frame
        ret, frame = self._cap.read()

        if not ret:
            self.close()
            raise StopIteration

        decode_ts = int(time.time() * 1000)

        # Create packet
        h, w = frame.shape[:2]

        packet = FramePacket(
            frame_id=self._frame_id,
            image=frame,
            width=w,
            height=h,
            recv_ts_ms=recv_ts,
            decode_ts_ms=decode_ts,
            pixel_format="BGR",
            source_id=self._path.name,
        )

        self._frame_id += 1

        return packet

    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_number: Target frame number

        Returns:
            True if seek successful
        """
        if self._closed:
            return False

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._frame_id = frame_number
        return True

    def seek_time(self, seconds: float) -> bool:
        """
        Seek to specific time.

        Args:
            seconds: Target time in seconds

        Returns:
            True if seek successful
        """
        frame_number = int(seconds * self._fps)
        return self.seek(frame_number)

    def close(self) -> None:
        """Release video capture resources."""
        if not self._closed:
            self._cap.release()
            self._closed = True

    def __repr__(self) -> str:
        return (
            f"FileIngest(path={self._path.name}, "
            f"fps={self._fps:.1f}, frames={self._frame_count})"
        )


class StubIngest(BaseIngest):
    """
    Stub ingest for testing.

    Generates dummy frames or returns provided frames.
    """

    def __init__(
        self,
        frames: Optional[list[np.ndarray]] = None,
        num_frames: int = 100,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ):
        """
        Initialize stub ingest.

        Args:
            frames: List of frames to return (generates random if None)
            num_frames: Number of frames to generate (if frames is None)
            width: Frame width
            height: Frame height
            fps: Frame rate
        """
        self._frames = frames
        self._num_frames = num_frames if frames is None else len(frames)
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_id = 0
        self._closed = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> Optional[int]:
        return self._num_frames

    def __iter__(self) -> Iterator[FramePacket]:
        return self

    def __next__(self) -> FramePacket:
        if self._closed or self._frame_id >= self._num_frames:
            raise StopIteration

        now_ms = int(time.time() * 1000)

        # Get or generate frame
        if self._frames is not None:
            image = self._frames[self._frame_id]
        else:
            image = np.random.randint(
                0, 256, (self._height, self._width, 3), dtype=np.uint8
            )

        h, w = image.shape[:2]

        packet = FramePacket(
            frame_id=self._frame_id,
            image=image,
            width=w,
            height=h,
            recv_ts_ms=now_ms,
            decode_ts_ms=now_ms,
            pixel_format="BGR",
            source_id="stub",
        )

        self._frame_id += 1
        return packet

    def close(self) -> None:
        self._closed = True
