"""
Intersection over Union (IoU) utilities.

Follows SRP: Only responsible for IoU calculations.
"""

from typing import List, Tuple

import numpy as np

from .bbox import BBoxXYXY


def compute_iou(box1: BBoxXYXY, box2: BBoxXYXY) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: First bbox (x1, y1, x2, y2)
        box2: Second bbox (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # IoU
    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_matrix(
    boxes1: List[BBoxXYXY], boxes2: List[BBoxXYXY]
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.

    Args:
        boxes1: First set of bboxes
        boxes2: Second set of bboxes

    Returns:
        IoU matrix of shape (len(boxes1), len(boxes2))
    """
    n1 = len(boxes1)
    n2 = len(boxes2)

    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    iou_matrix = np.zeros((n1, n2), dtype=np.float32)

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = compute_iou(box1, box2)

    return iou_matrix


def compute_iou_matrix_vectorized(
    boxes1: np.ndarray, boxes2: np.ndarray
) -> np.ndarray:
    """
    Compute IoU matrix using vectorized operations.

    More efficient for large number of boxes.

    Args:
        boxes1: Array of shape (N, 4) with xyxy format
        boxes2: Array of shape (M, 4) with xyxy format

    Returns:
        IoU matrix of shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Ensure float type
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection coordinates
    # boxes1: (N, 4), boxes2: (M, 4)
    # Expand to (N, M, 4) and (N, M, 4) for broadcasting
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union area
    union = area1[:, None] + area2[None, :] - intersection

    # IoU (avoid division by zero)
    iou = np.where(union > 0, intersection / union, 0)

    return iou.astype(np.float32)


def compute_giou(box1: BBoxXYXY, box2: BBoxXYXY) -> float:
    """
    Compute Generalized IoU between two bounding boxes.

    GIoU = IoU - (enclosing_area - union) / enclosing_area

    Args:
        box1: First bbox (x1, y1, x2, y2)
        box2: Second bbox (x1, y1, x2, y2)

    Returns:
        GIoU value between -1 and 1
    """
    # Standard IoU calculation
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return -1.0

    iou = intersection / union

    # Enclosing box
    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])

    enclosing_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    if enclosing_area <= 0:
        return iou

    giou = iou - (enclosing_area - union) / enclosing_area

    return giou


def compute_diou(box1: BBoxXYXY, box2: BBoxXYXY) -> float:
    """
    Compute Distance IoU between two bounding boxes.

    DIoU = IoU - (center_distance^2 / diagonal^2)

    Args:
        box1: First bbox (x1, y1, x2, y2)
        box2: Second bbox (x1, y1, x2, y2)

    Returns:
        DIoU value between -1 and 1
    """
    # Standard IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return -1.0

    iou = intersection / union

    # Center points
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    # Center distance squared
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal squared
    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])

    diagonal_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    if diagonal_sq <= 0:
        return iou

    diou = iou - center_dist_sq / diagonal_sq

    return diou


def nms(
    boxes: List[BBoxXYXY],
    scores: List[float],
    threshold: float = 0.45,
) -> List[int]:
    """
    Non-maximum suppression.

    Args:
        boxes: List of bboxes
        scores: Confidence scores
        threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if not boxes:
        return []

    # Convert to numpy
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)

    # Sort by score (descending)
    order = scores_np.argsort()[::-1]

    keep = []

    while len(order) > 0:
        # Keep highest scoring box
        i = order[0]
        keep.append(int(i))

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = np.array([
            compute_iou(tuple(boxes_np[i]), tuple(boxes_np[j]))
            for j in remaining
        ])

        # Keep boxes with IoU below threshold
        mask = ious < threshold
        order = remaining[mask]

    return keep
