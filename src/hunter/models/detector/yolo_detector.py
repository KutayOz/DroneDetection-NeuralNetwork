"""
YOLO11-based detector implementation.

Uses the ultralytics library for YOLO inference.
Supports YOLO11 variants: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Follows OCP: Can be configured for different YOLO variants without code changes.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

from .base import BaseDetector, DetectionOutput
from ...core.config import DetectorConfig
from ...core.exceptions import ModelError
from ...utils.checksum import compute_sha256

# Import ultralytics with error handling
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None


class YOLODetector(BaseDetector):
    """
    YOLO11 detector using ultralytics library.

    Supports:
    - YOLO11 PyTorch models (.pt)
    - ONNX exported models (.onnx)
    - TensorRT engines (.engine)

    Example usage:
        config = DetectorConfig(
            model_path=Path("yolo11m.pt"),
            confidence_threshold=0.5,
            device="cuda"
        )
        detector = YOLODetector(config)
        detections = detector.detect(image)
    """

    def __init__(self, config: DetectorConfig):
        """
        Initialize YOLO detector.

        Args:
            config: Detector configuration

        Raises:
            ImportError: If ultralytics is not installed
            ModelError: If model file not found or invalid
        """
        if not HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics package required for YOLODetector. "
                "Install with: pip install ultralytics"
            )

        self._config = config
        self._model_path = Path(config.model_path)

        # Validate model file exists
        if not self._model_path.exists():
            raise ModelError(f"Model file not found: {self._model_path}")

        # Compute model hash for versioning
        self._name = self._model_path.stem
        try:
            self._hash = compute_sha256(self._model_path)
        except Exception as e:
            raise ModelError(f"Failed to compute model hash: {e}")

        # Load model
        try:
            self._model = YOLO(str(self._model_path))
        except Exception as e:
            raise ModelError(f"Failed to load YOLO model: {e}")

        # Set device
        self._device = config.device
        if config.device == "cuda":
            try:
                self._model.to("cuda")
            except Exception:
                # Fall back to CPU if CUDA not available
                self._device = "cpu"
        elif config.device == "mps":
            try:
                self._model.to("mps")
            except Exception:
                self._device = "cpu"

        # Store inference parameters
        self._conf_threshold = config.confidence_threshold
        self._nms_threshold = config.nms_threshold
        self._max_detections = config.max_detections

        self._initialized = False

    @property
    def name(self) -> str:
        """Model name (filename without extension)."""
        return self._name

    @property
    def hash(self) -> str:
        """Model SHA256 hash."""
        return self._hash

    @property
    def device(self) -> str:
        """Device model is running on."""
        return self._device

    def detect(self, image: np.ndarray) -> List[DetectionOutput]:
        """
        Run YOLO detection on image.

        Args:
            image: Input image
                - BGR or RGB, uint8 (H, W, 3)
                - Or normalized float32

        Returns:
            List of DetectionOutput objects

        Note:
            YOLO handles preprocessing internally, so pass
            the original image without normalization.
        """
        # Run inference
        try:
            results = self._model(
                image,
                conf=self._conf_threshold,
                iou=self._nms_threshold,
                max_det=self._max_detections,
                verbose=False,
            )
        except Exception as e:
            raise ModelError(f"YOLO inference failed: {e}")

        # Parse results
        detections = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get bbox coordinates
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    detections.append(
                        DetectionOutput(
                            bbox_xyxy=(
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                            ),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectionOutput]]:
        """
        Run detection on batch of images.

        More efficient than individual calls for multiple images.

        Args:
            images: List of input images

        Returns:
            List of detection lists
        """
        if not images:
            return []

        try:
            results = self._model(
                images,
                conf=self._conf_threshold,
                iou=self._nms_threshold,
                max_det=self._max_detections,
                verbose=False,
            )
        except Exception as e:
            raise ModelError(f"YOLO batch inference failed: {e}")

        all_detections = []

        for result in results:
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    detections.append(
                        DetectionOutput(
                            bbox_xyxy=(
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                            ),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

            all_detections.append(detections)

        return all_detections

    def warmup(self, input_size: tuple = (640, 640)) -> None:
        """
        Warm up model with dummy inference.

        First inference is slow due to:
        - CUDA kernel compilation
        - Memory allocation
        - Model optimization

        Args:
            input_size: Size for dummy input (width, height)
        """
        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        _ = self.detect(dummy)
        self._initialized = True

    def get_model_info(self) -> dict:
        """
        Get detailed model information.

        Returns:
            Dictionary with model details
        """
        return {
            "name": self._name,
            "hash": self._hash,
            "hash_short": self._hash[:8],
            "path": str(self._model_path),
            "device": self._device,
            "conf_threshold": self._conf_threshold,
            "nms_threshold": self._nms_threshold,
            "max_detections": self._max_detections,
            "initialized": self._initialized,
        }


class StubDetector(BaseDetector):
    """
    Stub detector for testing.

    Returns configurable dummy detections.
    """

    def __init__(
        self,
        detections: Optional[List[DetectionOutput]] = None,
        name: str = "stub_detector",
    ):
        """
        Initialize stub detector.

        Args:
            detections: Fixed detections to return (empty if None)
            name: Model name
        """
        self._detections = detections or []
        self._name = name
        self._hash = "0" * 64  # Dummy hash

    @property
    def name(self) -> str:
        return self._name

    @property
    def hash(self) -> str:
        return self._hash

    def detect(self, image: np.ndarray) -> List[DetectionOutput]:
        """Return configured detections."""
        return self._detections.copy()

    def warmup(self) -> None:
        """No-op for stub."""
        pass

    def set_detections(self, detections: List[DetectionOutput]) -> None:
        """
        Set detections to return.

        Args:
            detections: Detections to return on next detect() call
        """
        self._detections = detections
