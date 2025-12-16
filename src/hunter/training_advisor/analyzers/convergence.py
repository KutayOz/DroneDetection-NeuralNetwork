"""
Convergence analyzer.

Detects convergence issues like plateaus in training metrics.
"""

from typing import List

from .base import BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class ConvergenceAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting convergence issues.

    Detects:
    - Plateau: loss not improving for many epochs
    - Underfitting: high loss even after many epochs
    """

    def __init__(
        self,
        plateau_epochs: int = 5,
        min_improvement: float = 0.001,
        underfitting_threshold: float = 0.5,
        underfitting_min_epochs: int = 20,
    ):
        """
        Initialize convergence analyzer.

        Args:
            plateau_epochs: Consecutive epochs with no improvement to flag
            min_improvement: Minimum loss decrease to count as improvement
            underfitting_threshold: Loss threshold for underfitting
            underfitting_min_epochs: Minimum epochs before checking underfitting
        """
        self._plateau_epochs = plateau_epochs
        self._min_improvement = min_improvement
        self._underfitting_threshold = underfitting_threshold
        self._underfitting_min_epochs = underfitting_min_epochs

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "convergence"

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze for convergence issues.

        Args:
            metrics: Training metrics

        Returns:
            List of detected issues
        """
        issues = []

        if len(metrics.epochs) < self._plateau_epochs:
            return issues

        # Check for plateau
        issues.extend(self._check_plateau(metrics))

        # Check for underfitting
        if len(metrics.epochs) >= self._underfitting_min_epochs:
            issues.extend(self._check_underfitting(metrics))

        return issues

    def _check_plateau(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for training plateau."""
        issues = []

        recent = metrics.epochs[-self._plateau_epochs:]
        losses = [e.val_loss for e in recent]

        # Check if loss is not improving
        max_loss = max(losses)
        min_loss = min(losses)
        improvement = max_loss - min_loss

        if improvement < self._min_improvement:
            issues.append(Issue(
                issue_type=IssueType.PLATEAU,
                severity=IssueSeverity.MEDIUM,
                message=f"Training plateaued: only {improvement:.6f} improvement over {self._plateau_epochs} epochs",
                details={
                    "improvement": improvement,
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "epochs_checked": self._plateau_epochs,
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues

    def _check_underfitting(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for underfitting (high loss after many epochs)."""
        issues = []

        current_loss = metrics.latest_epoch.val_loss if metrics.latest_epoch else 0

        if current_loss > self._underfitting_threshold:
            issues.append(Issue(
                issue_type=IssueType.UNDERFITTING,
                severity=IssueSeverity.MEDIUM,
                message=f"Possible underfitting: val_loss={current_loss:.4f} after {len(metrics.epochs)} epochs",
                details={
                    "current_loss": current_loss,
                    "threshold": self._underfitting_threshold,
                    "epochs": len(metrics.epochs),
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues
