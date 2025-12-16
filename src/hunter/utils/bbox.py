"""
Bounding box utilities.

Follows SRP: Only responsible for bbox operations.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


# Type aliases for bbox formats
BBoxXYXY = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
BBoxXYWH = Tuple[float, float, float, float]  # (x, y, width, height)
BBoxCXCYWH = Tuple[float, float, float, float]  # (center_x, center_y, width, height)


@dataclass(frozen=True)
class BBox:
    """
    Immutable bounding box representation.

    Internally stores as (x1, y1, x2, y2) format.
    Provides conversion methods to other formats.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        """Validate bbox coordinates."""
        if self.x2 < self.x1 or self.y2 < self.y1:
            # Swap if needed (defensive)
            object.__setattr__(self, "x1", min(self.x1, self.x2))
            object.__setattr__(self, "y1", min(self.y1, self.y2))
            object.__setattr__(self, "x2", max(self.x1, self.x2))
            object.__setattr__(self, "y2", max(self.y1, self.y2))

    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def center_x(self) -> float:
        """Center x coordinate."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Center y coordinate."""
        return (self.y1 + self.y2) / 2

    def as_xyxy(self) -> BBoxXYXY:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> BBoxXYWH:
        """Return as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width, self.height)

    def as_cxcywh(self) -> BBoxCXCYWH:
        """Return as (center_x, center_y, width, height) tuple."""
        return (self.center_x, self.center_y, self.width, self.height)

    def as_int_xyxy(self) -> Tuple[int, int, int, int]:
        """Return as integer (x1, y1, x2, y2) tuple."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def clip(self, width: int, height: int) -> "BBox":
        """
        Clip bbox to image boundaries.

        Args:
            width: Image width
            height: Image height

        Returns:
            Clipped BBox
        """
        return BBox(
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
        )

    def scale(self, scale_x: float, scale_y: float) -> "BBox":
        """
        Scale bbox coordinates.

        Args:
            scale_x: X scale factor
            scale_y: Y scale factor

        Returns:
            Scaled BBox
        """
        return BBox(
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
            x2=self.x2 * scale_x,
            y2=self.y2 * scale_y,
        )

    def pad(self, padding: float) -> "BBox":
        """
        Expand bbox by padding.

        Args:
            padding: Padding amount (positive = expand, negative = shrink)

        Returns:
            Padded BBox
        """
        return BBox(
            x1=self.x1 - padding,
            y1=self.y1 - padding,
            x2=self.x2 + padding,
            y2=self.y2 + padding,
        )

    def pad_percent(self, percent: float) -> "BBox":
        """
        Expand bbox by percentage of its size.

        Args:
            percent: Padding percentage (0.1 = 10% expansion)

        Returns:
            Padded BBox
        """
        pad_x = self.width * percent / 2
        pad_y = self.height * percent / 2
        return BBox(
            x1=self.x1 - pad_x,
            y1=self.y1 - pad_y,
            x2=self.x2 + pad_x,
            y2=self.y2 + pad_y,
        )

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        """Create from (x1, y1, x2, y2) format."""
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        """Create from (x, y, width, height) format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> "BBox":
        """Create from (center_x, center_y, width, height) format."""
        return cls(
            x1=cx - w / 2,
            y1=cy - h / 2,
            x2=cx + w / 2,
            y2=cy + h / 2,
        )

    @classmethod
    def from_tuple(cls, bbox: Union[BBoxXYXY, Tuple]) -> "BBox":
        """Create from tuple (assumes xyxy format)."""
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])


def xyxy_to_xywh(bbox: BBoxXYXY) -> BBoxXYWH:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(bbox: BBoxXYWH) -> BBoxXYXY:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def xyxy_to_cxcywh(bbox: BBoxXYXY) -> BBoxCXCYWH:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return (cx, cy, w, h)


def cxcywh_to_xyxy(bbox: BBoxCXCYWH) -> BBoxXYXY:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = bbox
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def clip_bbox(
    bbox: BBoxXYXY, width: int, height: int
) -> BBoxXYXY:
    """
    Clip bbox to image boundaries.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        width: Image width
        height: Image height

    Returns:
        Clipped bounding box
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def scale_bbox(
    bbox: BBoxXYXY, scale_x: float, scale_y: float
) -> BBoxXYXY:
    """
    Scale bbox coordinates.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        scale_x: X scale factor
        scale_y: Y scale factor

    Returns:
        Scaled bounding box
    """
    x1, y1, x2, y2 = bbox
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def bbox_area(bbox: BBoxXYXY) -> float:
    """Calculate bounding box area."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: BBoxXYXY) -> Tuple[float, float]:
    """Get bounding box center point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
