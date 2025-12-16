"""
Unit tests for utility functions.
"""

import hashlib
import numpy as np
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from hunter.utils.iou import (
    compute_iou,
    compute_iou_matrix,
    compute_giou,
)
from hunter.utils.bbox import (
    BBox,
    BBoxXYXY,
    xyxy_to_xywh,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    clip_bbox,
    scale_bbox,
    bbox_area,
    bbox_center,
)
from hunter.utils.checksum import (
    compute_sha256,
    compute_checksum_short,
    verify_checksum,
)


# ============================================
# IoU Tests
# ============================================


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes have IoU of 1.0."""
        box1: BBoxXYXY = (0, 0, 100, 100)
        box2: BBoxXYXY = (0, 0, 100, 100)
        assert compute_iou(box1, box2) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes have IoU of 0.0."""
        box1: BBoxXYXY = (0, 0, 50, 50)
        box2: BBoxXYXY = (100, 100, 150, 150)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes have IoU between 0 and 1."""
        box1: BBoxXYXY = (0, 0, 100, 100)
        box2: BBoxXYXY = (50, 50, 150, 150)
        iou = compute_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Expected: intersection = 50*50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 = 0.1429
        assert iou == pytest.approx(2500 / 17500, rel=0.01)

    def test_contained_box(self):
        """Contained box has IoU = small_area / large_area."""
        box1: BBoxXYXY = (0, 0, 100, 100)  # area = 10000
        box2: BBoxXYXY = (25, 25, 75, 75)  # area = 2500
        iou = compute_iou(box1, box2)
        # intersection = 2500, union = 10000
        assert iou == pytest.approx(2500 / 10000, rel=0.01)

    def test_symmetric(self):
        """IoU is symmetric."""
        box1: BBoxXYXY = (0, 0, 100, 100)
        box2: BBoxXYXY = (50, 50, 150, 150)
        assert compute_iou(box1, box2) == compute_iou(box2, box1)


class TestComputeIoUMatrix:
    """Tests for IoU matrix computation."""

    def test_square_matrix(self):
        """IoU matrix with same number of boxes."""
        boxes1: list[BBoxXYXY] = [
            (0, 0, 100, 100),
            (200, 200, 300, 300),
        ]
        boxes2: list[BBoxXYXY] = [
            (0, 0, 100, 100),
            (200, 200, 300, 300),
        ]
        matrix = compute_iou_matrix(boxes1, boxes2)
        assert matrix.shape == (2, 2)
        # Diagonal should be 1.0
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)
        # Off-diagonal should be 0.0 (no overlap)
        assert matrix[0, 1] == pytest.approx(0.0)
        assert matrix[1, 0] == pytest.approx(0.0)

    def test_rectangular_matrix(self):
        """IoU matrix with different number of boxes."""
        boxes1: list[BBoxXYXY] = [(0, 0, 100, 100)]
        boxes2: list[BBoxXYXY] = [
            (0, 0, 100, 100),
            (50, 50, 150, 150),
        ]
        matrix = compute_iou_matrix(boxes1, boxes2)
        assert matrix.shape == (1, 2)


class TestComputeGIoU:
    """Tests for Generalized IoU."""

    def test_identical_boxes(self):
        """Identical boxes have GIoU of 1.0."""
        box1: BBoxXYXY = (0, 0, 100, 100)
        box2: BBoxXYXY = (0, 0, 100, 100)
        assert compute_giou(box1, box2) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping boxes have GIoU < 0."""
        box1: BBoxXYXY = (0, 0, 50, 50)
        box2: BBoxXYXY = (100, 100, 150, 150)
        giou = compute_giou(box1, box2)
        # GIoU can be negative for non-overlapping boxes
        assert giou < 0.0

    def test_giou_range(self):
        """GIoU is in range [-1, 1]."""
        box1: BBoxXYXY = (0, 0, 100, 100)
        box2: BBoxXYXY = (50, 50, 150, 150)
        giou = compute_giou(box1, box2)
        assert -1.0 <= giou <= 1.0


# ============================================
# Bbox Conversion Tests
# ============================================


class TestBboxConversions:
    """Tests for bbox format conversions."""

    def test_xyxy_to_xywh(self):
        """Convert xyxy to xywh format."""
        xyxy: BBoxXYXY = (10, 20, 110, 120)
        x, y, w, h = xyxy_to_xywh(xyxy)
        assert x == 10
        assert y == 20
        assert w == 100
        assert h == 100

    def test_xywh_to_xyxy(self):
        """Convert xywh to xyxy format."""
        xywh = (10, 20, 100, 100)
        x1, y1, x2, y2 = xywh_to_xyxy(xywh)
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110
        assert y2 == 120

    def test_xyxy_to_cxcywh(self):
        """Convert xyxy to center format."""
        xyxy: BBoxXYXY = (0, 0, 100, 100)
        cx, cy, w, h = xyxy_to_cxcywh(xyxy)
        assert cx == 50
        assert cy == 50
        assert w == 100
        assert h == 100

    def test_cxcywh_to_xyxy(self):
        """Convert center format to xyxy."""
        cxcywh = (50, 50, 100, 100)
        x1, y1, x2, y2 = cxcywh_to_xyxy(cxcywh)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 100

    def test_roundtrip_xyxy_xywh(self):
        """xyxy -> xywh -> xyxy roundtrip."""
        original: BBoxXYXY = (10, 20, 110, 120)
        xywh = xyxy_to_xywh(original)
        result = xywh_to_xyxy(xywh)
        assert result == original

    def test_roundtrip_xyxy_cxcywh(self):
        """xyxy -> cxcywh -> xyxy roundtrip."""
        original: BBoxXYXY = (0, 0, 100, 100)
        cxcywh = xyxy_to_cxcywh(original)
        result = cxcywh_to_xyxy(cxcywh)
        assert result == original


class TestBboxOperations:
    """Tests for bbox operations."""

    def test_clip_bbox(self):
        """Clip bbox to image bounds."""
        bbox: BBoxXYXY = (-10, -10, 110, 110)
        clipped = clip_bbox(bbox, width=100, height=100)
        assert clipped == (0, 0, 100, 100)

    def test_clip_bbox_no_change(self):
        """No change when bbox within bounds."""
        bbox: BBoxXYXY = (10, 10, 90, 90)
        clipped = clip_bbox(bbox, width=100, height=100)
        assert clipped == bbox

    def test_scale_bbox(self):
        """Scale bbox by factor."""
        bbox: BBoxXYXY = (10, 10, 50, 50)
        scaled = scale_bbox(bbox, scale_x=2.0, scale_y=2.0)
        assert scaled == (20, 20, 100, 100)

    def test_bbox_area(self):
        """Compute bbox area."""
        assert bbox_area((0, 0, 10, 10)) == 100
        assert bbox_area((0, 0, 10, 20)) == 200

    def test_bbox_center(self):
        """Compute bbox center."""
        cx, cy = bbox_center((0, 0, 100, 100))
        assert cx == 50
        assert cy == 50


class TestBBoxClass:
    """Tests for BBox dataclass."""

    def test_creation(self):
        """BBox can be created."""
        bbox = BBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_properties(self):
        """BBox properties work correctly."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.area == 10000
        assert bbox.center == (50, 50)

    def test_from_xywh(self):
        """BBox.from_xywh creates correctly."""
        bbox = BBox.from_xywh(10, 20, 100, 100)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 120


# ============================================
# Checksum Tests
# ============================================


class TestChecksum:
    """Tests for checksum utilities."""

    def test_compute_sha256(self):
        """Compute hash of file."""
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            hash_value = compute_sha256(path)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64  # SHA256 hex
        finally:
            path.unlink()

    def test_compute_sha256_deterministic(self):
        """Same data produces same hash."""
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            hash1 = compute_sha256(path)
            hash2 = compute_sha256(path)
            assert hash1 == hash2
        finally:
            path.unlink()

    def test_compute_checksum_short(self):
        """Compute short hash."""
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"model weights" * 1000)
            path = Path(f.name)

        try:
            hash_value = compute_checksum_short(path, length=16)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 16
        finally:
            path.unlink()

    def test_verify_checksum(self):
        """Verify file matches expected hash."""
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            expected = compute_sha256(path)
            assert verify_checksum(path, expected)
            assert not verify_checksum(path, "wrong_hash")
        finally:
            path.unlink()

    def test_compute_sha256_nonexistent(self):
        """Raise error for nonexistent file."""
        from hunter.core.exceptions import ModelError
        with pytest.raises(ModelError):
            compute_sha256(Path("/nonexistent/path/file.txt"))
