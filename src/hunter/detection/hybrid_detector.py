"""
Hybrid detector with confidence-based routing.

Combines primary detector (YOLO) with secondary verifier (Siamese)
using confidence thresholds for intelligent routing.

Follows Open/Closed Principle (OCP):
- Open for extension (new detectors/verifiers can be injected)
- Closed for modification (routing logic doesn't change)

Follows Dependency Inversion Principle (DIP):
- Depends on IDetector/IVerifier abstractions
- Concrete implementations injected at runtime
"""

from typing import List, Optional

import cv2
import numpy as np

from ..interfaces.detector import (
    IDetector,
    IVerifier,
    IDetectionRouter,
    BoundingBox,
    DetectionResult,
)


class HybridDetector:
    """
    Hybrid detection pipeline with confidence-based routing.

    Routes detections based on confidence:
    - High confidence (>high_threshold) → Direct pass
    - Medium confidence (low_threshold-high_threshold) → Verify with Siamese
    - Low confidence (<low_threshold) → Discard

    Implements both IDetector (for detection) and IDetectionRouter (for routing).
    """

    def __init__(
        self,
        primary_detector: IDetector,
        verifier: IVerifier,
        high_threshold: float = 0.8,
        low_threshold: float = 0.4,
        crop_padding: float = 0.1,
    ) -> None:
        """
        Initialize hybrid detector.

        Args:
            primary_detector: Primary object detector (e.g., YOLO11)
            verifier: Secondary verifier (e.g., SiameseVerifier)
            high_threshold: Confidence threshold for direct pass
            low_threshold: Confidence threshold below which to discard
            crop_padding: Padding factor when cropping detections for verification
        """
        self._primary_detector = primary_detector
        self._verifier = verifier
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._crop_padding = crop_padding

        # Validate thresholds
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError(
                f"Invalid thresholds: low={low_threshold}, high={high_threshold}. "
                f"Must satisfy: 0 <= low < high <= 1"
            )

    @property
    def name(self) -> str:
        """Model identifier name."""
        return f"HybridDetector({self._primary_detector.name}+{self._verifier.name})"

    @property
    def model_hash(self) -> str:
        """Combined model hash."""
        return f"{self._primary_detector.model_hash}_{self._verifier.model_hash}"

    @property
    def high_confidence_threshold(self) -> float:
        """Threshold for direct pass."""
        return self._high_threshold

    @property
    def low_confidence_threshold(self) -> float:
        """Threshold below which to discard."""
        return self._low_threshold

    def _crop_detection(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:
        """
        Crop detection region from image with padding.

        Args:
            image: Source image (HWC)
            bbox: Detection bounding box

        Returns:
            Cropped region
        """
        h, w = image.shape[:2]

        # Apply padding
        padded_bbox = bbox.pad(self._crop_padding)

        # Clip to image boundaries
        x1 = int(max(0, padded_bbox.x1))
        y1 = int(max(0, padded_bbox.y1))
        x2 = int(min(w, padded_bbox.x2))
        y2 = int(min(h, padded_bbox.y2))

        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            # Return minimal crop if bbox is invalid
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return image[y1:y2, x1:x2].copy()

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects with hybrid routing.

        Args:
            image: Input image (HWC format)

        Returns:
            List of verified/filtered DetectionResult objects
        """
        # Run primary detector
        raw_detections = self._primary_detector.detect(image)

        # Route detections
        return self.route(raw_detections, image)

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
        if len(detections) == 0:
            return []

        # Categorize detections
        high_conf: List[DetectionResult] = []
        medium_conf: List[DetectionResult] = []
        crops_to_verify: List[np.ndarray] = []

        for det in detections:
            if det.confidence >= self._high_threshold:
                # High confidence - direct pass
                high_conf.append(self._add_routing_metadata(det, "direct"))
            elif det.confidence >= self._low_threshold:
                # Medium confidence - needs verification
                medium_conf.append(det)
                crops_to_verify.append(self._crop_detection(image, det.bbox))
            # Low confidence - implicitly discarded

        # Verify medium confidence detections
        verified: List[DetectionResult] = []
        if len(medium_conf) > 0:
            verification_results = self._verifier.verify_batch(crops_to_verify)

            for det, result in zip(medium_conf, verification_results):
                if result.is_verified:
                    # Passed verification - add with embedding
                    verified_det = DetectionResult(
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_id=det.class_id,
                        embedding=result.embedding,
                        metadata={
                            **(det.metadata or {}),
                            "routing_path": "siamese",
                            "verification_score": result.similarity_score,
                        },
                    )
                    verified.append(verified_det)

        # Combine results
        return high_conf + verified

    def _add_routing_metadata(
        self,
        detection: DetectionResult,
        routing_path: str,
    ) -> DetectionResult:
        """
        Add routing metadata to detection.

        Args:
            detection: Original detection
            routing_path: Routing path taken

        Returns:
            Detection with added metadata
        """
        return DetectionResult(
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            embedding=detection.embedding,
            metadata={
                **(detection.metadata or {}),
                "routing_path": routing_path,
            },
        )

    def warmup(self) -> None:
        """Warm up both detector and verifier."""
        self._primary_detector.warmup()
        self._verifier.warmup()

    def update_thresholds(
        self,
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None,
    ) -> None:
        """
        Update routing thresholds.

        Args:
            high_threshold: New high confidence threshold
            low_threshold: New low confidence threshold
        """
        new_high = high_threshold if high_threshold is not None else self._high_threshold
        new_low = low_threshold if low_threshold is not None else self._low_threshold

        if not 0.0 <= new_low < new_high <= 1.0:
            raise ValueError(
                f"Invalid thresholds: low={new_low}, high={new_high}. "
                f"Must satisfy: 0 <= low < high <= 1"
            )

        self._high_threshold = new_high
        self._low_threshold = new_low

    def get_routing_stats(
        self,
        detections: List[DetectionResult],
    ) -> dict:
        """
        Get routing statistics for detections (without actually routing).

        Args:
            detections: Detections to analyze

        Returns:
            Dict with counts for each routing path
        """
        stats = {
            "total": len(detections),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
        }

        for det in detections:
            if det.confidence >= self._high_threshold:
                stats["high_confidence"] += 1
            elif det.confidence >= self._low_threshold:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1

        return stats
