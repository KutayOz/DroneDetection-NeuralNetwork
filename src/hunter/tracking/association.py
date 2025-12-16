"""
Detection-Track association using Hungarian algorithm.

Combines IoU cost and embedding cost for optimal matching.

Follows SRP: Only responsible for association logic.
Follows OCP: Cost functions can be extended without modifying core logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.iou import compute_iou


@dataclass
class Detection:
    """
    Single detection from the detector.

    Represents a detected object in a single frame.
    """

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get detection center point."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Get detection area."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class AssociationResult:
    """
    Result of association step.

    Contains matched pairs and unmatched indices.
    """

    matched_pairs: List[Tuple[int, int]]  # (track_idx, detection_idx)
    unmatched_tracks: List[int]
    unmatched_detections: List[int]

    @property
    def n_matches(self) -> int:
        """Number of matches."""
        return len(self.matched_pairs)

    @property
    def n_unmatched_tracks(self) -> int:
        """Number of unmatched tracks."""
        return len(self.unmatched_tracks)

    @property
    def n_unmatched_detections(self) -> int:
        """Number of unmatched detections."""
        return len(self.unmatched_detections)


@dataclass
class AssociationConfig:
    """Configuration for association algorithm."""

    iou_threshold: float = 0.3  # Minimum IoU for valid match
    embedding_weight: float = 0.3  # Weight for embedding cost (0-1)
    gate_threshold: float = 0.1  # Minimum IoU to even consider matching
    max_distance: float = 1e6  # Cost for impossible matches

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= self.embedding_weight <= 1:
            raise ValueError("embedding_weight must be in [0, 1]")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("gate_threshold must be in [0, 1]")


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine distance between embeddings.

    Assumes embeddings are L2-normalized.

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Distance in [0, 2] (0 = identical, 2 = opposite)
    """
    similarity = np.dot(emb1, emb2)
    return 1.0 - similarity


def compute_cost_matrix(
    track_boxes: List[Tuple[float, float, float, float]],
    track_embeddings: List[Optional[np.ndarray]],
    detections: List[Detection],
    config: AssociationConfig,
) -> np.ndarray:
    """
    Compute cost matrix for Hungarian algorithm.

    Cost = (1 - w) * iou_cost + w * embedding_cost

    Args:
        track_boxes: Predicted bbox for each track
        track_embeddings: Embedding for each track (can be None)
        detections: Detections from current frame
        config: Association configuration

    Returns:
        Cost matrix of shape (n_tracks, n_detections)
    """
    n_tracks = len(track_boxes)
    n_dets = len(detections)

    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    # Initialize with max distance
    cost_matrix = np.full((n_tracks, n_dets), config.max_distance, dtype=np.float32)

    for t_idx in range(n_tracks):
        t_box = track_boxes[t_idx]
        t_emb = track_embeddings[t_idx]

        for d_idx in range(n_dets):
            det = detections[d_idx]

            # Compute IoU
            iou = compute_iou(t_box, det.bbox_xyxy)

            # Gate: skip if IoU too low
            if iou < config.gate_threshold:
                continue

            # IoU cost (lower is better)
            iou_cost = 1.0 - iou

            # Embedding cost (if available)
            emb_cost = 0.0
            if (
                t_emb is not None
                and det.embedding is not None
                and config.embedding_weight > 0
            ):
                emb_cost = compute_embedding_distance(t_emb, det.embedding)

            # Combined cost
            w = config.embedding_weight
            cost = (1 - w) * iou_cost + w * emb_cost

            cost_matrix[t_idx, d_idx] = cost

    return cost_matrix


class HungarianAssociator:
    """
    Hungarian algorithm based track-detection association.

    Uses scipy's linear_sum_assignment for optimal matching.
    Combines IoU and embedding costs.
    """

    def __init__(self, config: Optional[AssociationConfig] = None):
        """
        Initialize associator.

        Args:
            config: Configuration (uses defaults if None)
        """
        self._config = config or AssociationConfig()

    def associate(
        self,
        track_boxes: List[Tuple[float, float, float, float]],
        track_embeddings: List[Optional[np.ndarray]],
        detections: List[Detection],
    ) -> AssociationResult:
        """
        Associate tracks with detections.

        Uses Hungarian algorithm to find optimal assignment
        that minimizes total cost.

        Args:
            track_boxes: Predicted bbox for each track
            track_embeddings: Embedding for each track
            detections: Detections from current frame

        Returns:
            AssociationResult with matches and unmatched indices
        """
        n_tracks = len(track_boxes)
        n_dets = len(detections)

        # Handle empty cases
        if n_tracks == 0:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=[],
                unmatched_detections=list(range(n_dets)),
            )

        if n_dets == 0:
            return AssociationResult(
                matched_pairs=[],
                unmatched_tracks=list(range(n_tracks)),
                unmatched_detections=[],
            )

        # Compute cost matrix
        cost_matrix = compute_cost_matrix(
            track_boxes, track_embeddings, detections, self._config
        )

        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by threshold
        matched_pairs = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_dets))

        # Threshold: cost should be less than (1 - iou_threshold)
        threshold = 1.0 - self._config.iou_threshold

        for t_idx, d_idx in zip(row_indices, col_indices):
            cost = cost_matrix[t_idx, d_idx]

            if cost < threshold and cost < self._config.max_distance:
                matched_pairs.append((int(t_idx), int(d_idx)))
                unmatched_tracks.discard(t_idx)
                unmatched_detections.discard(d_idx)

        return AssociationResult(
            matched_pairs=matched_pairs,
            unmatched_tracks=sorted(unmatched_tracks),
            unmatched_detections=sorted(unmatched_detections),
        )

    def associate_cascade(
        self,
        tracks_by_age: List[List[Tuple[int, Tuple[float, float, float, float], Optional[np.ndarray]]]],
        detections: List[Detection],
    ) -> AssociationResult:
        """
        Cascade association by track age.

        Associates younger (more reliable) tracks first,
        then older tracks with remaining detections.

        Args:
            tracks_by_age: List of track groups, sorted by age (youngest first)
                          Each track: (original_idx, bbox, embedding)
            detections: Detections from current frame

        Returns:
            AssociationResult with global indices
        """
        all_matched = []
        remaining_dets = list(range(len(detections)))
        unmatched_track_indices = []

        for track_group in tracks_by_age:
            if not remaining_dets:
                # No more detections, all remaining tracks are unmatched
                unmatched_track_indices.extend([t[0] for t in track_group])
                continue

            # Extract data for this group
            original_indices = [t[0] for t in track_group]
            boxes = [t[1] for t in track_group]
            embeddings = [t[2] for t in track_group]

            # Get remaining detections
            remaining_det_list = [detections[i] for i in remaining_dets]

            # Associate
            result = self.associate(boxes, embeddings, remaining_det_list)

            # Map back to original indices
            for local_t_idx, local_d_idx in result.matched_pairs:
                original_t_idx = original_indices[local_t_idx]
                original_d_idx = remaining_dets[local_d_idx]
                all_matched.append((original_t_idx, original_d_idx))

            # Update remaining detections
            matched_d_indices = {remaining_dets[d] for _, d in result.matched_pairs}
            remaining_dets = [d for d in remaining_dets if d not in matched_d_indices]

            # Track unmatched tracks
            unmatched_track_indices.extend(
                [original_indices[i] for i in result.unmatched_tracks]
            )

        return AssociationResult(
            matched_pairs=all_matched,
            unmatched_tracks=sorted(unmatched_track_indices),
            unmatched_detections=sorted(remaining_dets),
        )


class IoUOnlyAssociator:
    """
    Simple IoU-only associator.

    For cases where embeddings are not available.
    """

    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize associator.

        Args:
            iou_threshold: Minimum IoU for valid match
        """
        self._iou_threshold = iou_threshold

    def associate(
        self,
        track_boxes: List[Tuple[float, float, float, float]],
        detections: List[Detection],
    ) -> AssociationResult:
        """
        Associate using IoU only.

        Args:
            track_boxes: Predicted bbox for each track
            detections: Detections from current frame

        Returns:
            AssociationResult
        """
        config = AssociationConfig(
            iou_threshold=self._iou_threshold,
            embedding_weight=0.0,  # No embedding
            gate_threshold=0.0,
        )

        associator = HungarianAssociator(config)
        return associator.associate(
            track_boxes,
            [None] * len(track_boxes),  # No embeddings
            detections,
        )
