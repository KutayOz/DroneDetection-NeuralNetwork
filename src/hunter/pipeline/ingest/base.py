"""
Abstract video ingest interface.

Follows OCP: New ingest sources can be added without modifying existing code.
Follows ISP: Minimal interface for all video sources.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .frame_packet import FramePacket


class BaseIngest(ABC):
    """
    Abstract base class for video ingest.

    All video source implementations must inherit from this class.
    Provides iterator interface for frame-by-frame processing.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[FramePacket]:
        """
        Get iterator for frames.

        Returns:
            Iterator yielding FramePacket objects
        """
        pass

    @abstractmethod
    def __next__(self) -> FramePacket:
        """
        Get next frame.

        Returns:
            Next FramePacket

        Raises:
            StopIteration: When no more frames available
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Release resources.

        Should be called when done with the source.
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """
        Source frame rate.

        Returns:
            Frames per second
        """
        pass

    @property
    @abstractmethod
    def frame_count(self) -> Optional[int]:
        """
        Total number of frames.

        Returns:
            Frame count or None if unknown (live streams)
        """
        pass

    @property
    def is_live(self) -> bool:
        """
        Whether source is a live stream.

        Returns:
            True if live stream (frame_count is None)
        """
        return self.frame_count is None

    def __enter__(self) -> "BaseIngest":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close resources."""
        self.close()
