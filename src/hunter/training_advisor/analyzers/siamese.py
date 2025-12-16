"""
Siamese analyzer.

Detects issues specific to Siamese embedding training.
"""

from typing import List

from .base import BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class SiameseAnalyzer(BaseAnalyzer):
    """
    Analyzer for Siamese/embedding training issues.

    Detects:
    - Triplet loss stuck
    - Embedding not converging
    """

    def __init__(
        self,
        stuck_epochs: int = 10,
        stuck_threshold: float = 0.01,
        convergence_threshold: float = 0.001,
    ):
        """
        Initialize Siamese analyzer.

        Args:
            stuck_epochs: Epochs to check for stuck loss
            stuck_threshold: Minimum change to not be considered stuck
            convergence_threshold: Threshold for convergence detection
        """
        self._stuck_epochs = stuck_epochs
        self._stuck_threshold = stuck_threshold
        self._convergence_threshold = convergence_threshold

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "siamese"

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze for Siamese training issues.

        Args:
            metrics: Training metrics

        Returns:
            List of detected issues
        """
        issues = []

        # Skip if no embedding metrics
        if not metrics.has_embedding_metrics:
            return issues

        if len(metrics.epochs) < self._stuck_epochs:
            return issues

        # Check for stuck triplet loss
        issues.extend(self._check_triplet_loss_stuck(metrics))

        # Check for embedding convergence
        issues.extend(self._check_embedding_convergence(metrics))

        return issues

    def _check_triplet_loss_stuck(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for stuck triplet loss."""
        issues = []

        recent = metrics.epochs[-self._stuck_epochs:]
        triplet_losses = [e.triplet_loss for e in recent if e.triplet_loss is not None]

        if len(triplet_losses) < 2:
            return issues

        max_loss = max(triplet_losses)
        min_loss = min(triplet_losses)
        change = max_loss - min_loss

        if change < self._stuck_threshold:
            issues.append(Issue(
                issue_type=IssueType.TRIPLET_LOSS_STUCK,
                severity=IssueSeverity.MEDIUM,
                message=f"Triplet loss stuck: only {change:.6f} change over {self._stuck_epochs} epochs",
                details={
                    "change": change,
                    "threshold": self._stuck_threshold,
                    "recent_losses": triplet_losses,
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues

    def _check_embedding_convergence(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for embedding convergence issues."""
        issues = []

        recent = metrics.epochs[-self._stuck_epochs:]
        embedding_losses = [e.embedding_loss for e in recent if e.embedding_loss is not None]

        if len(embedding_losses) < 2:
            return issues

        # Check if embedding loss is not improving
        first_loss = embedding_losses[0]
        last_loss = embedding_losses[-1]
        improvement = first_loss - last_loss

        if improvement < self._convergence_threshold and len(embedding_losses) >= self._stuck_epochs:
            issues.append(Issue(
                issue_type=IssueType.EMBEDDING_NOT_CONVERGING,
                severity=IssueSeverity.MEDIUM,
                message=f"Embedding not converging: {improvement:.6f} improvement",
                details={
                    "improvement": improvement,
                    "threshold": self._convergence_threshold,
                    "first_loss": first_loss,
                    "last_loss": last_loss,
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues
