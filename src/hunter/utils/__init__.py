"""
Utility modules.

Provides common utilities:
- Bounding box operations
- IoU calculations
- Checksum verification
"""

from .bbox import (
    BBox,
    BBoxCXCYWH,
    BBoxXYWH,
    BBoxXYXY,
    bbox_area,
    bbox_center,
    clip_bbox,
    cxcywh_to_xyxy,
    scale_bbox,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
)
from .checksum import (
    ChecksumRegistry,
    compute_checksum_short,
    compute_md5,
    compute_sha256,
    get_checksum_registry,
    verify_checksum,
)
from .iou import (
    compute_diou,
    compute_giou,
    compute_iou,
    compute_iou_matrix,
    compute_iou_matrix_vectorized,
    nms,
)

__all__ = [
    # BBox
    "BBox",
    "BBoxXYXY",
    "BBoxXYWH",
    "BBoxCXCYWH",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "clip_bbox",
    "scale_bbox",
    "bbox_area",
    "bbox_center",
    # IoU
    "compute_iou",
    "compute_iou_matrix",
    "compute_iou_matrix_vectorized",
    "compute_giou",
    "compute_diou",
    "nms",
    # Checksum
    "compute_sha256",
    "compute_md5",
    "verify_checksum",
    "compute_checksum_short",
    "ChecksumRegistry",
    "get_checksum_registry",
]
