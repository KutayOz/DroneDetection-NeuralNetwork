"""
Overfitting analyzer.

Detects overfitting patterns in training metrics.
"""

from typing import List

from .base import BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class OverfittingAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting overfitting.

    Detects when training loss decreases but validation loss increases,
    indicating the model is memorizing training data rather than learning
    generalizable patterns.
    """

    def __init__(
        self,
        gap_threshold: float = 0.1,
        min_epochs: int = 3,
        increasing_val_epochs: int = 3,
    ):
        """
        Initialize overfitting analyzer.

        Args:
            gap_threshold: Minimum train-val loss gap to flag
            min_epochs: Minimum epochs before checking
            increasing_val_epochs: Consecutive epochs of val loss increase
        """
        self._gap_threshold = gap_threshold
        self._min_epochs = min_epochs
        self._increasing_val_epochs = increasing_val_epochs

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "overfitting"

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze for overfitting patterns.

        Args:
            metrics: Training metrics

        Returns:
            List of detected overfitting issues
        """
        issues = []

        if len(metrics.epochs) < self._min_epochs:
            return issues

        # Check for increasing gap
        issues.extend(self._check_gap_trend(metrics))

        # Check for val loss increasing while train loss decreases
        issues.extend(self._check_diverging_losses(metrics))

        return issues

    def _check_gap_trend(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for increasing train-val gap."""
        issues = []
        recent = metrics.epochs[-self._min_epochs:]

        # Calculate gap for recent epochs
        gaps = [e.loss_gap for e in recent]

        # Check if gap is increasing and above threshold
        if len(gaps) >= 2:
            gap_increasing = all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1))
            current_gap = gaps[-1]

            if gap_increasing and current_gap > self._gap_threshold:
                severity = self._determine_severity(current_gap)
                issues.append(Issue(
                    issue_type=IssueType.OVERFITTING,
                    severity=severity,
                    message=f"Train-val gap increasing: {current_gap:.4f} (threshold: {self._gap_threshold})",
                    details={
                        "current_gap": current_gap,
                        "threshold": self._gap_threshold,
                        "recent_gaps": gaps,
                    },
                    epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
                ))

        return issues

    def _check_diverging_losses(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for train loss decreasing while val loss increases."""
        issues = []

        if len(metrics.epochs) < self._increasing_val_epochs + 1:
            return issues

        recent = metrics.epochs[-(self._increasing_val_epochs + 1):]

        # Check train loss trend (should be decreasing)
        train_decreasing = all(
            recent[i].train_loss > recent[i + 1].train_loss
            for i in range(len(recent) - 1)
        )

        # Check val loss trend (increasing = overfitting)
        val_increasing = all(
            recent[i].val_loss < recent[i + 1].val_loss
            for i in range(len(recent) - 1)
        )

        if train_decreasing and val_increasing:
            current_gap = metrics.latest_epoch.loss_gap if metrics.latest_epoch else 0
            severity = self._determine_severity(current_gap)

            issues.append(Issue(
                issue_type=IssueType.OVERFITTING,
                severity=severity,
                message=f"Model overfitting: train loss decreasing but val loss increasing for {self._increasing_val_epochs} epochs",
                details={
                    "train_losses": [e.train_loss for e in recent],
                    "val_losses": [e.val_loss for e in recent],
                    "epochs_detected": self._increasing_val_epochs,
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues

    def _determine_severity(self, gap: float) -> IssueSeverity:
        """Determine issue severity based on gap size."""
        if gap > 0.5:
            return IssueSeverity.CRITICAL
        elif gap > 0.3:
            return IssueSeverity.HIGH
        elif gap > 0.15:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
