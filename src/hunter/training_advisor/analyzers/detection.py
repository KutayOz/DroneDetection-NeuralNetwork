"""
Detection analyzer.

Detects issues specific to object detection training (YOLO).
"""

from typing import List

from .base import BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class DetectionAnalyzer(BaseAnalyzer):
    """
    Analyzer for object detection training issues.

    Detects:
    - Low mAP: detection accuracy below threshold
    - Precision-recall imbalance
    - Class imbalance indicators
    """

    def __init__(
        self,
        low_map_threshold: float = 0.3,
        imbalance_threshold: float = 0.3,
        min_epochs: int = 5,
    ):
        """
        Initialize detection analyzer.

        Args:
            low_map_threshold: Minimum acceptable mAP@50
            imbalance_threshold: Threshold for precision-recall imbalance
            min_epochs: Minimum epochs before checking
        """
        self._low_map_threshold = low_map_threshold
        self._imbalance_threshold = imbalance_threshold
        self._min_epochs = min_epochs

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "detection"

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze for detection-specific issues.

        Args:
            metrics: Training metrics

        Returns:
            List of detected issues
        """
        issues = []

        # Skip if no detection metrics
        if not metrics.has_detection_metrics:
            return issues

        if len(metrics.epochs) < self._min_epochs:
            return issues

        # Check for low mAP
        issues.extend(self._check_low_map(metrics))

        # Check for precision-recall imbalance
        issues.extend(self._check_precision_recall(metrics))

        return issues

    def _check_low_map(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for low mAP scores."""
        issues = []

        latest = metrics.latest_epoch
        if not latest or latest.map50 is None:
            return issues

        if latest.map50 < self._low_map_threshold:
            severity = self._determine_map_severity(latest.map50)
            issues.append(Issue(
                issue_type=IssueType.LOW_MAP,
                severity=severity,
                message=f"Low mAP@50: {latest.map50:.3f} (threshold: {self._low_map_threshold})",
                details={
                    "map50": latest.map50,
                    "map50_95": latest.map50_95,
                    "threshold": self._low_map_threshold,
                },
                epoch_detected=latest.epoch,
            ))

        return issues

    def _check_precision_recall(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for precision-recall imbalance."""
        issues = []

        latest = metrics.latest_epoch
        if not latest or latest.precision is None or latest.recall is None:
            return issues

        imbalance = abs(latest.precision - latest.recall)

        if imbalance > self._imbalance_threshold:
            if latest.precision > latest.recall:
                message = f"High precision ({latest.precision:.3f}) but low recall ({latest.recall:.3f})"
                detail = "Model is too conservative - missing many detections"
            else:
                message = f"High recall ({latest.recall:.3f}) but low precision ({latest.precision:.3f})"
                detail = "Model has too many false positives"

            issues.append(Issue(
                issue_type=IssueType.PRECISION_RECALL_IMBALANCE,
                severity=IssueSeverity.MEDIUM,
                message=message,
                details={
                    "precision": latest.precision,
                    "recall": latest.recall,
                    "imbalance": imbalance,
                    "suggestion": detail,
                },
                epoch_detected=latest.epoch,
            ))

        return issues

    def _determine_map_severity(self, map50: float) -> IssueSeverity:
        """Determine severity based on mAP value."""
        if map50 < 0.1:
            return IssueSeverity.CRITICAL
        elif map50 < 0.2:
            return IssueSeverity.HIGH
        elif map50 < 0.3:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
