"""
Base interface definitions using Python Protocol classes.

These interfaces define the contracts that all implementations must follow.
Using Protocol (structural subtyping) instead of ABC for flexibility.
"""

from typing import Protocol, List, Optional, Iterator, Tuple, Any
from dataclasses import dataclass
import numpy as np


# ============================================
# Data Transfer Objects (DTOs)
# ============================================

@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectionResult:
    """Result from a detector."""
    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


@dataclass
class FrameData:
    """Input frame data."""
    frame_id: int
    image: np.ndarray
    timestamp_ms: int
    width: int
    height: int
    source_id: str = "default"


@dataclass
class TrackData:
    """Track state data."""
    track_id: int
    bbox: BoundingBox
    predicted_bbox: BoundingBox
    confidence: float
    state: str
    velocity: Tuple[float, float]
    age_frames: int
    hits: int
    time_since_update: int


# ============================================
# Detection Interfaces
# ============================================

class IDetector(Protocol):
    """Interface for object detectors."""

    @property
    def name(self) -> str:
        """Model name."""
        ...

    @property
    def model_hash(self) -> str:
        """Model checksum for tracking."""
        ...

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in image.

        Args:
            image: Input image (RGB, HWC format)

        Returns:
            List of detection results
        """
        ...

    def warmup(self) -> None:
        """Warm up model for consistent latency."""
        ...


class IDetectionRouter(Protocol):
    """Interface for hybrid detection routing."""

    def route(
        self,
        detections: List[DetectionResult],
        image: np.ndarray,
    ) -> List[DetectionResult]:
        """
        Route detections through verification pipeline.

        High confidence -> direct pass
        Medium confidence -> secondary verification
        Low confidence -> discard

        Args:
            detections: Raw detections from primary detector
            image: Original image for cropping

        Returns:
            Verified detections
        """
        ...


# ============================================
# Embedding Interfaces
# ============================================

class IEmbedder(Protocol):
    """Interface for appearance embedding."""

    @property
    def name(self) -> str:
        """Model name."""
        ...

    @property
    def model_hash(self) -> str:
        """Model checksum."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of embedding vector."""
        ...

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for cropped detection.

        Args:
            crop: Cropped detection image

        Returns:
            Normalized embedding vector
        """
        ...

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Batch embedding for efficiency."""
        ...


class ISimilarityScorer(Protocol):
    """Interface for computing embedding similarity."""

    def score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute similarity between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        ...

    def score_batch(
        self,
        query: np.ndarray,
        gallery: List[np.ndarray],
    ) -> List[float]:
        """Batch similarity scoring."""
        ...


# ============================================
# Tracking Interfaces
# ============================================

class IStateMachine(Protocol):
    """Interface for track state machine."""

    @property
    def state(self) -> str:
        """Current state name."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether track is still active."""
        ...

    @property
    def is_visible(self) -> bool:
        """Whether track should be in output."""
        ...

    def update(self, matched: bool) -> Tuple[str, Optional[str]]:
        """
        Update state machine.

        Args:
            matched: Whether detection matched this frame

        Returns:
            (new_state, transition_reason) - reason is None if no transition
        """
        ...


class IAssociator(Protocol):
    """Interface for detection-track association."""

    def associate(
        self,
        tracks: List[TrackData],
        detections: List[DetectionResult],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks.

        Args:
            tracks: Current tracks
            detections: Current frame detections

        Returns:
            (matched_pairs, unmatched_tracks, unmatched_detections)
            matched_pairs: List of (track_idx, detection_idx)
        """
        ...


class ITracker(Protocol):
    """Interface for multi-object tracker."""

    def update(
        self,
        detections: List[DetectionResult],
        timestamp_ms: int,
    ) -> List[TrackData]:
        """
        Update tracker with new detections.

        Args:
            detections: Current frame detections
            timestamp_ms: Frame timestamp

        Returns:
            List of active tracks
        """
        ...

    def get_visible_tracks(self) -> List[TrackData]:
        """Get tracks that should be in output."""
        ...

    def reset(self) -> None:
        """Reset tracker state."""
        ...


# ============================================
# Pipeline Interfaces
# ============================================

class IIngest(Protocol):
    """Interface for video input."""

    @property
    def fps(self) -> float:
        """Source frame rate."""
        ...

    @property
    def frame_count(self) -> int:
        """Total frames (0 if unknown/live)."""
        ...

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over frames."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class IPreprocessor(Protocol):
    """Interface for image preprocessing."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Raw input image

        Returns:
            Preprocessed image
        """
        ...


class IOutputSink(Protocol):
    """Interface for output writing."""

    def write(self, message: Any) -> None:
        """Write output message."""
        ...

    def close(self) -> None:
        """Close output sink."""
        ...
