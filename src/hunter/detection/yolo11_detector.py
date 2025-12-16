"""
YOLO11 detector implementation.

Primary detector using Ultralytics YOLO v11 for object detection.

Follows Single Responsibility Principle (SRP):
- Only handles YOLO inference
- Configuration via constructor injection
- No verification logic (handled by HybridDetector)
"""

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from ..interfaces.detector import (
    IDetector,
    BoundingBox,
    DetectionResult,
)
from ..utils.checksum import compute_checksum_short


def _lazy_import_yolo():
    """Lazy import YOLO to avoid import overhead when not needed."""
    from ultralytics import YOLO
    return YOLO


class YOLO11Detector:
    """
    YOLO v11 object detector implementation.

    Implements IDetector protocol for integration with hybrid detection pipeline.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        half_precision: bool = False,
        target_classes: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize YOLO11 detector.

        Args:
            model_path: Path to YOLO model file (.pt, .onnx, .engine)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression IoU threshold
            max_detections: Maximum number of detections to return
            device: Inference device ("cuda", "cpu", "mps")
            half_precision: Use FP16 inference (GPU only)
            target_classes: Filter to specific class IDs (None = all classes)
        """
        self._model_path = Path(model_path)
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._max_detections = max_detections
        self._device = device
        self._half_precision = half_precision
        self._target_classes = target_classes

        # Lazy load model
        self._model = None
        self._model_hash_cached: Optional[str] = None

    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._model is None:
            YOLO = _lazy_import_yolo()
            self._model = YOLO(str(self._model_path))
            # Move to device
            if self._device != "cpu":
                self._model.to(self._device)

    @property
    def name(self) -> str:
        """Model identifier name."""
        return f"YOLO11-{self._model_path.stem}"

    @property
    def model_hash(self) -> str:
        """Model file checksum for tracking."""
        if self._model_hash_cached is None:
            if self._model_path.exists():
                self._model_hash_cached = compute_checksum_short(
                    self._model_path, length=12
                )
            else:
                self._model_hash_cached = "model_not_found"
        return self._model_hash_cached

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in image.

        Args:
            image: Input image as numpy array (HWC format, BGR or RGB)

        Returns:
            List of DetectionResult objects above confidence threshold
        """
        self._ensure_model_loaded()

        # Run inference
        results = self._model(
            image,
            conf=self._confidence_threshold,
            iou=self._nms_threshold,
            max_det=self._max_detections,
            verbose=False,
            half=self._half_precision,
            classes=self._target_classes,
        )

        # Parse results
        detections: List[DetectionResult] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            # Get arrays
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls

            for i in range(len(xyxy)):
                # Filter by confidence (should already be filtered by YOLO)
                if conf[i] < self._confidence_threshold:
                    continue

                bbox = BoundingBox(
                    x1=float(xyxy[i][0]),
                    y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]),
                    y2=float(xyxy[i][3]),
                )

                detection = DetectionResult(
                    bbox=bbox,
                    confidence=float(conf[i]),
                    class_id=int(cls[i]),
                    metadata={"detector": self.name},
                )
                detections.append(detection)

        return detections

    def warmup(self) -> None:
        """
        Warm up model with dummy inference.

        First inference is typically slower due to CUDA kernel compilation
        and memory allocation. Call this before timing-critical inference.
        """
        self._ensure_model_loaded()

        # Create dummy image
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)

    @property
    def confidence_threshold(self) -> float:
        """Current confidence threshold."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Update confidence threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence threshold must be in [0, 1], got {value}")
        self._confidence_threshold = value

    @property
    def nms_threshold(self) -> float:
        """Current NMS threshold."""
        return self._nms_threshold

    @nms_threshold.setter
    def nms_threshold(self, value: float) -> None:
        """Update NMS threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"NMS threshold must be in [0, 1], got {value}")
        self._nms_threshold = value
