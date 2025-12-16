"""
Pipeline interfaces using Python Protocol classes.

Defines contracts for:
- IIngest: Video input source
- IPreprocessor: Image preprocessing
- IOutputSink: Result output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np

from .tracker import TrackSnapshot


# ============================================
# Data Transfer Objects
# ============================================


@dataclass
class FramePacket:
    """
    Input frame data packet.

    Contains raw image data with metadata for tracking through pipeline.
    """

    frame_id: int
    image: np.ndarray  # HWC format
    timestamp_ms: int
    width: int
    height: int
    source_id: str = "default"
    pixel_format: str = "BGR"  # OpenCV default
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        """Image shape (H, W, C)."""
        return self.image.shape

    @property
    def effective_timestamp(self) -> int:
        """Timestamp to use for tracking."""
        return self.timestamp_ms

    @classmethod
    def from_numpy(
        cls,
        frame_id: int,
        image: np.ndarray,
        source_id: str = "default",
        timestamp_ms: Optional[int] = None,
    ) -> FramePacket:
        """Create FramePacket from numpy array."""
        import time

        h, w = image.shape[:2]
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        return cls(
            frame_id=frame_id,
            image=image,
            timestamp_ms=ts,
            width=w,
            height=h,
            source_id=source_id,
        )


@dataclass
class ModelInfo:
    """Model identification for output messages."""

    detector_name: str
    detector_hash: str
    embedder_name: Optional[str] = None
    embedder_hash: Optional[str] = None
    verifier_name: Optional[str] = None
    verifier_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_name": self.detector_name,
            "detector_hash": self.detector_hash,
            "embedder_name": self.embedder_name,
            "embedder_hash": self.embedder_hash,
            "verifier_name": self.verifier_name,
            "verifier_hash": self.verifier_hash,
        }


@dataclass
class PipelineMetrics:
    """Per-frame pipeline timing metrics."""

    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    verify_ms: float = 0.0
    embed_ms: float = 0.0
    associate_ms: float = 0.0
    output_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Total pipeline latency."""
        return (
            self.preprocess_ms
            + self.detect_ms
            + self.verify_ms
            + self.embed_ms
            + self.associate_ms
            + self.output_ms
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "detect_ms": round(self.detect_ms, 2),
            "verify_ms": round(self.verify_ms, 2),
            "embed_ms": round(self.embed_ms, 2),
            "associate_ms": round(self.associate_ms, 2),
            "output_ms": round(self.output_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class TrackMessage:
    """
    Output message containing tracking results.

    Serializable to JSON for downstream consumers.
    """

    frame_id: int
    timestamp_ms: int
    model_info: ModelInfo
    metrics: PipelineMetrics
    tracks: List[TrackSnapshot] = field(default_factory=list)
    detection_count: int = 0

    # Class constant (not a dataclass field)
    MSG_VERSION: str = field(default="1.0", init=False, repr=False)

    @property
    def track_count(self) -> int:
        """Number of tracks in message."""
        return len(self.tracks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "msg_version": self.MSG_VERSION,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "model": self.model_info.to_dict(),
            "metrics": self.metrics.to_dict(),
            "detection_count": self.detection_count,
            "track_count": self.track_count,
            "tracks": [self._track_to_dict(t) for t in self.tracks],
        }

    def _track_to_dict(self, track: TrackSnapshot) -> Dict[str, Any]:
        """Convert track to dictionary."""
        return {
            "track_id": track.track_id,
            "state": track.state_name,
            "confidence": round(track.confidence, 3),
            "bbox_xyxy": list(track.bbox.as_int_tuple()),
            "predicted_bbox_xyxy": list(track.predicted_bbox.as_int_tuple()),
            "velocity_px_per_s": [round(v, 2) for v in track.velocity],
            "age_frames": track.age_frames,
            "hits": track.hits,
            "time_since_update": track.time_since_update,
            "trajectory": [
                {"t_ms": p.timestamp_ms, "cx": round(p.cx, 1), "cy": round(p.cy, 1)}
                for p in track.trajectory[-10:]  # Last 10 points
            ],
        }


# ============================================
# Protocol Interfaces
# ============================================


@runtime_checkable
class IIngest(Protocol):
    """
    Interface for video input sources.

    Implementations: FileIngest, RTSPIngest, StubIngest
    """

    @property
    def fps(self) -> float:
        """Source frame rate."""
        ...

    @property
    def frame_count(self) -> int:
        """Total frames (0 if unknown/live stream)."""
        ...

    @property
    def width(self) -> int:
        """Frame width."""
        ...

    @property
    def height(self) -> int:
        """Frame height."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether source is open and readable."""
        ...

    def __iter__(self) -> Iterator[FramePacket]:
        """Iterate over frames."""
        ...

    def read(self) -> Optional[FramePacket]:
        """Read single frame (None if no more frames)."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


@runtime_checkable
class IPreprocessor(Protocol):
    """
    Interface for image preprocessing.

    Implementations: Preprocessor, LetterboxPreprocessor
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Raw input image (HWC, BGR)

        Returns:
            Preprocessed image ready for model
        """
        ...

    def process_for_detector(self, image: np.ndarray) -> np.ndarray:
        """Preprocess specifically for detector input."""
        ...

    def process_for_embedder(
        self, image: np.ndarray, bbox: Any
    ) -> np.ndarray:
        """
        Crop and preprocess for embedder input.

        Args:
            image: Full frame
            bbox: Bounding box to crop

        Returns:
            Preprocessed crop
        """
        ...


@runtime_checkable
class IOutputSink(Protocol):
    """
    Interface for output destinations.

    Implementations: JsonSink, StubSink, UDPSink
    """

    def write(self, message: TrackMessage) -> None:
        """
        Write tracking message to output.

        Args:
            message: TrackMessage to write
        """
        ...

    def flush(self) -> None:
        """Flush any buffered output."""
        ...

    def close(self) -> None:
        """Close output sink and release resources."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether sink is open and writable."""
        ...
