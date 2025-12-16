"""
Detector interfaces using Python Protocol classes.

Follows Interface Segregation Principle (ISP):
- Small, focused interfaces
- Clients only depend on methods they use

Follows Dependency Inversion Principle (DIP):
- High-level modules depend on abstractions
- Low-level modules implement abstractions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


# ============================================
# Value Objects (Immutable Data Structures)
# ============================================


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box in xyxy format.

    Coordinates are in pixel space (float for sub-pixel precision).
    Follows Value Object pattern - compared by value, not identity.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Validate coordinates."""
        if self.x1 > self.x2:
            object.__setattr__(self, "x1", self.x2)
            object.__setattr__(self, "x2", self.x1)
        if self.y1 > self.y2:
            object.__setattr__(self, "y1", self.y2)
            object.__setattr__(self, "y2", self.y1)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Box area in pixels^2."""
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """Return as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width, self.height)

    def as_cxcywh(self) -> Tuple[float, float, float, float]:
        """Return as (center_x, center_y, width, height) tuple."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        """Return as integer tuple (for pixel indexing)."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def scale(self, sx: float, sy: float) -> BoundingBox:
        """Scale box by factors."""
        return BoundingBox(
            x1=self.x1 * sx,
            y1=self.y1 * sy,
            x2=self.x2 * sx,
            y2=self.y2 * sy,
        )

    def translate(self, dx: float, dy: float) -> BoundingBox:
        """Translate box by offset."""
        return BoundingBox(
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )

    def pad(self, padding: float) -> BoundingBox:
        """Expand box by padding (relative to size)."""
        pad_w = self.width * padding
        pad_h = self.height * padding
        return BoundingBox(
            x1=self.x1 - pad_w,
            y1=self.y1 - pad_h,
            x2=self.x2 + pad_w,
            y2=self.y2 + pad_h,
        )

    def clip(self, width: int, height: int) -> BoundingBox:
        """Clip box to image boundaries."""
        return BoundingBox(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
        )

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        """Create from (x, y, width, height) format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        """Create from (center_x, center_y, width, height) format."""
        return cls(
            x1=cx - w / 2,
            y1=cy - h / 2,
            x2=cx + w / 2,
            y2=cy + h / 2,
        )


@dataclass
class DetectionResult:
    """
    Result from an object detector.

    Contains bounding box, confidence, class ID, and optional embedding.
    """

    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"Class ID must be >= 0, got {self.class_id}")


@dataclass
class VerificationResult:
    """
    Result from a secondary verifier (Siamese network).

    Contains verification decision, similarity score, and optional embedding.
    """

    is_verified: bool
    similarity_score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"Similarity score must be in [0, 1], got {self.similarity_score}"
            )


# ============================================
# Protocol Interfaces
# ============================================


@runtime_checkable
class IDetector(Protocol):
    """
    Interface for object detectors.

    Implementations: YOLO11Detector, ONNXDetector, StubDetector
    """

    @property
    def name(self) -> str:
        """Model identifier name."""
        ...

    @property
    def model_hash(self) -> str:
        """Model file checksum for tracking."""
        ...

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in image.

        Args:
            image: Input image as numpy array (HWC format, RGB or BGR)

        Returns:
            List of DetectionResult objects

        Note:
            - Image format (RGB/BGR) depends on implementation
            - Confidence threshold applied internally
            - NMS applied internally
        """
        ...

    def warmup(self) -> None:
        """
        Warm up model with dummy inference.

        First inference is typically slower due to:
        - CUDA kernel compilation
        - Memory allocation
        - Graph optimization

        Call this before timing-critical inference.
        """
        ...


@runtime_checkable
class IEmbedder(Protocol):
    """
    Interface for appearance embedding models.

    Used for re-identification: comparing if two crops show the same object.
    Implementations: SiameseEmbedder, OSNetEmbedder, StubEmbedder
    """

    @property
    def name(self) -> str:
        """Model identifier name."""
        ...

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embedding vector."""
        ...

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for a single crop.

        Args:
            crop: Cropped detection image (HWC format)

        Returns:
            Normalized embedding vector (L2 norm = 1)
        """
        ...

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple crops (batched for efficiency).

        Args:
            crops: List of cropped images

        Returns:
            List of normalized embedding vectors
        """
        ...

    def warmup(self) -> None:
        """Warm up model."""
        ...


@runtime_checkable
class IVerifier(Protocol):
    """
    Interface for secondary verification models.

    Used in hybrid detection: verify uncertain detections from primary detector.
    Implementations: SiameseVerifier, StubVerifier
    """

    @property
    def name(self) -> str:
        """Model identifier name."""
        ...

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        ...

    @property
    def similarity_threshold(self) -> float:
        """Threshold for positive verification."""
        ...

    def verify(
        self,
        crop: np.ndarray,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Verify if crop contains target object.

        Args:
            crop: Cropped image to verify
            reference_embedding: Optional reference for comparison

        Returns:
            VerificationResult with decision and score
        """
        ...

    def verify_batch(
        self,
        crops: List[np.ndarray],
        reference_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[VerificationResult]:
        """
        Verify multiple crops (batched for efficiency).

        Args:
            crops: List of cropped images
            reference_embeddings: Optional list of references

        Returns:
            List of VerificationResult objects
        """
        ...

    def warmup(self) -> None:
        """Warm up model."""
        ...


@runtime_checkable
class IDetectionRouter(Protocol):
    """
    Interface for hybrid detection routing.

    Routes detections based on confidence:
    - High confidence -> direct pass
    - Medium confidence -> secondary verification
    - Low confidence -> discard
    """

    def route(
        self,
        detections: List[DetectionResult],
        image: np.ndarray,
    ) -> List[DetectionResult]:
        """
        Route detections through verification pipeline.

        Args:
            detections: Raw detections from primary detector
            image: Original image for cropping

        Returns:
            Verified/filtered detections
        """
        ...

    @property
    def high_confidence_threshold(self) -> float:
        """Threshold for direct pass."""
        ...

    @property
    def low_confidence_threshold(self) -> float:
        """Threshold below which to discard."""
        ...
