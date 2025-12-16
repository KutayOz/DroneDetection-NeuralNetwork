"""
Image preprocessing transforms.

Prepares images for model input.
"""

from typing import Tuple

import cv2
import numpy as np

from ...core.config import PreprocessConfig


class Preprocessor:
    """
    Image preprocessor for detector input.

    Performs:
    - Color space conversion (BGR ↔ RGB)
    - Resize to model input size
    - Normalization

    Note: For YOLO, minimal preprocessing is needed as the
    model handles it internally. This is mainly for custom models.
    """

    def __init__(self, config: PreprocessConfig):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self._config = config
        self._input_size = config.input_size  # (width, height)
        self._mean = np.array(config.normalize_mean, dtype=np.float32)
        self._std = np.array(config.normalize_std, dtype=np.float32)
        self._target_format = config.pixel_format

    @property
    def input_size(self) -> Tuple[int, int]:
        """Target input size (width, height)."""
        return self._input_size

    def process(
        self,
        image: np.ndarray,
        source_format: str = "BGR",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image (H, W, 3), uint8
            source_format: Input pixel format ("BGR" or "RGB")
            normalize: Whether to apply normalization

        Returns:
            Preprocessed image (H, W, 3), float32 if normalized
        """
        result = image

        # Color conversion if needed
        if source_format != self._target_format:
            if source_format == "BGR" and self._target_format == "RGB":
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            elif source_format == "RGB" and self._target_format == "BGR":
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Resize
        result = cv2.resize(
            result,
            self._input_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize
        if normalize:
            result = result.astype(np.float32) / 255.0
            result = (result - self._mean) / self._std

        return result

    def process_for_yolo(
        self,
        image: np.ndarray,
        source_format: str = "BGR",
    ) -> np.ndarray:
        """
        Minimal preprocessing for YOLO.

        YOLO handles most preprocessing internally,
        so we just do color conversion if needed.

        Args:
            image: Input image
            source_format: Input pixel format

        Returns:
            Image ready for YOLO (uint8)
        """
        result = image

        # YOLO expects RGB
        if source_format == "BGR":
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result

    def get_scale_factors(
        self,
        original_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        Get scale factors for coordinate mapping.

        Used to map detection coordinates back to original image.

        Args:
            original_size: (width, height) of original image

        Returns:
            (scale_x, scale_y) to multiply with detection coords
        """
        orig_w, orig_h = original_size
        new_w, new_h = self._input_size
        return (orig_w / new_w, orig_h / new_h)

    def scale_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        original_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        """
        Scale bbox from model output to original image coordinates.

        Args:
            bbox: Bbox in model coordinate space (x1, y1, x2, y2)
            original_size: Original image size (width, height)

        Returns:
            Bbox in original coordinate space
        """
        scale_x, scale_y = self.get_scale_factors(original_size)
        x1, y1, x2, y2 = bbox
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing (preserve aspect ratio).

    Args:
        image: Input image (H, W, 3)
        target_size: Target size (width, height)
        color: Padding color

    Returns:
        Tuple of:
        - Letterboxed image
        - Scale factor
        - Padding (dw, dh)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    # Create output with padding
    output = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    output[dh : dh + new_h, dw : dw + new_w] = resized

    return output, scale, (dw, dh)


def crop_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: float = 0.0,
) -> np.ndarray:
    """
    Crop image region defined by bbox.

    Args:
        image: Input image (H, W, 3)
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Fractional padding to add (0.1 = 10%)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    if padding > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 = x1 - bw * padding / 2
        y1 = y1 - bh * padding / 2
        x2 = x2 + bw * padding / 2
        y2 = y2 + bh * padding / 2

    # Clip to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # Ensure valid crop
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=image.dtype)

    return image[y1:y2, x1:x2].copy()
