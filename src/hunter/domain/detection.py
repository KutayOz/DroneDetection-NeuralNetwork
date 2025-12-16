"""
Detection domain entities.

Contains business logic for detection results and batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..interfaces.detector import BoundingBox


@dataclass
class Detection:
    """
    Domain entity for a single detection.

    Encapsulates detection data with business methods.
    """

    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate detection."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def has_embedding(self) -> bool:
        """Whether detection has appearance embedding."""
        return self.embedding is not None

    @property
    def center(self) -> Tuple[float, float]:
        """Detection center point."""
        return self.bbox.center

    @property
    def area(self) -> float:
        """Detection area in pixels^2."""
        return self.bbox.area

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        if self.bbox.height == 0:
            return 0.0
        return self.bbox.width / self.bbox.height

    def with_embedding(self, embedding: np.ndarray) -> Detection:
        """Return new Detection with embedding."""
        return Detection(
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            embedding=embedding,
            metadata=self.metadata.copy(),
        )


@dataclass
class VerifiedDetection(Detection):
    """
    Detection that has passed through verification pipeline.

    Extends Detection with verification metadata.
    """

    is_verified: bool = False
    verification_score: float = 0.0
    routing_path: str = "unknown"  # "direct", "siamese", "discarded"

    @classmethod
    def from_detection(
        cls,
        detection: Detection,
        is_verified: bool,
        verification_score: float,
        routing_path: str,
    ) -> VerifiedDetection:
        """Create VerifiedDetection from Detection."""
        return cls(
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            embedding=detection.embedding,
            metadata=detection.metadata,
            is_verified=is_verified,
            verification_score=verification_score,
            routing_path=routing_path,
        )


@dataclass
class DetectionBatch:
    """
    Batch of detections from a single frame.

    Provides collection operations for detections.
    """

    frame_id: int
    timestamp_ms: int
    detections: List[Detection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    @property
    def is_empty(self) -> bool:
        """Whether batch has no detections."""
        return len(self.detections) == 0

    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)

    @property
    def bboxes(self) -> List[BoundingBox]:
        """All bounding boxes."""
        return [d.bbox for d in self.detections]

    @property
    def confidences(self) -> List[float]:
        """All confidence scores."""
        return [d.confidence for d in self.detections]

    @property
    def embeddings(self) -> List[Optional[np.ndarray]]:
        """All embeddings (may include None)."""
        return [d.embedding for d in self.detections]

    def filter_by_confidence(self, threshold: float) -> DetectionBatch:
        """Return new batch with detections above threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionBatch(
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            detections=filtered,
        )

    def filter_by_class(self, class_ids: List[int]) -> DetectionBatch:
        """Return new batch with only specified classes."""
        filtered = [d for d in self.detections if d.class_id in class_ids]
        return DetectionBatch(
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            detections=filtered,
        )

    def sort_by_confidence(self, descending: bool = True) -> DetectionBatch:
        """Return new batch sorted by confidence."""
        sorted_dets = sorted(
            self.detections,
            key=lambda d: d.confidence,
            reverse=descending,
        )
        return DetectionBatch(
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            detections=sorted_dets,
        )

    def top_k(self, k: int) -> DetectionBatch:
        """Return top-k detections by confidence."""
        sorted_batch = self.sort_by_confidence()
        return DetectionBatch(
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            detections=sorted_batch.detections[:k],
        )
