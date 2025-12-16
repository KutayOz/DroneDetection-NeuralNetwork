"""
Learning rate analyzer.

Detects learning rate issues in training metrics.
"""

from typing import List

from .base import BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class LearningRateAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting learning rate issues.

    Detects:
    - LR too high: diverging loss, oscillations
    - LR too low: very slow convergence
    - Divergence: loss exploding
    """

    def __init__(
        self,
        divergence_threshold: float = 2.0,
        oscillation_threshold: float = 0.2,
        min_epochs_for_slow: int = 10,
        slow_improvement_threshold: float = 0.1,
    ):
        """
        Initialize learning rate analyzer.

        Args:
            divergence_threshold: Loss increase ratio to flag divergence
            oscillation_threshold: Variance threshold for oscillation
            min_epochs_for_slow: Minimum epochs before checking slow learning
            slow_improvement_threshold: Minimum improvement expected
        """
        self._divergence_threshold = divergence_threshold
        self._oscillation_threshold = oscillation_threshold
        self._min_epochs_for_slow = min_epochs_for_slow
        self._slow_improvement_threshold = slow_improvement_threshold

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "learning_rate"

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze for learning rate issues.

        Args:
            metrics: Training metrics

        Returns:
            List of detected issues
        """
        issues = []

        if len(metrics.epochs) < 2:
            return issues

        # Check for divergence
        issues.extend(self._check_divergence(metrics))

        # Check for oscillations
        issues.extend(self._check_oscillations(metrics))

        # Check for slow learning
        if len(metrics.epochs) >= self._min_epochs_for_slow:
            issues.extend(self._check_slow_learning(metrics))

        return issues

    def _check_divergence(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for diverging loss (LR too high)."""
        issues = []

        recent = metrics.epochs[-3:] if len(metrics.epochs) >= 3 else metrics.epochs

        # Check if loss is increasing
        loss_increasing = all(
            recent[i].val_loss < recent[i + 1].val_loss
            for i in range(len(recent) - 1)
        )

        if loss_increasing and len(recent) >= 2:
            ratio = recent[-1].val_loss / recent[0].val_loss
            if ratio > self._divergence_threshold:
                issues.append(Issue(
                    issue_type=IssueType.DIVERGENCE,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Loss diverging: increased by {ratio:.2f}x",
                    details={
                        "loss_ratio": ratio,
                        "start_loss": recent[0].val_loss,
                        "current_loss": recent[-1].val_loss,
                    },
                    epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
                ))

                issues.append(Issue(
                    issue_type=IssueType.LR_TOO_HIGH,
                    severity=IssueSeverity.HIGH,
                    message="Learning rate likely too high - loss is diverging",
                    details={"loss_ratio": ratio},
                    epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
                ))

        return issues

    def _check_oscillations(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for oscillating loss."""
        issues = []

        if len(metrics.epochs) < 5:
            return issues

        recent = metrics.epochs[-5:]
        losses = [e.val_loss for e in recent]

        # Calculate variance
        mean_loss = sum(losses) / len(losses)
        variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        std = variance ** 0.5

        # Check for high relative variance
        if mean_loss > 0 and std / mean_loss > self._oscillation_threshold:
            issues.append(Issue(
                issue_type=IssueType.LR_TOO_HIGH,
                severity=IssueSeverity.MEDIUM,
                message=f"Loss oscillating: std/mean = {std / mean_loss:.3f}",
                details={
                    "std": std,
                    "mean": mean_loss,
                    "relative_variance": std / mean_loss,
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues

    def _check_slow_learning(self, metrics: TrainingMetrics) -> List[Issue]:
        """Check for very slow learning (LR too low)."""
        issues = []

        first_loss = metrics.epochs[0].val_loss
        current_loss = metrics.latest_epoch.val_loss if metrics.latest_epoch else first_loss

        # Calculate improvement
        improvement = (first_loss - current_loss) / first_loss if first_loss > 0 else 0

        # Check if improvement is too slow
        expected_improvement = self._slow_improvement_threshold * (len(metrics.epochs) / self._min_epochs_for_slow)

        if improvement < expected_improvement and improvement >= 0:
            issues.append(Issue(
                issue_type=IssueType.LR_TOO_LOW,
                severity=IssueSeverity.MEDIUM,
                message=f"Learning too slow: {improvement:.1%} improvement over {len(metrics.epochs)} epochs",
                details={
                    "improvement": improvement,
                    "expected": expected_improvement,
                    "epochs": len(metrics.epochs),
                },
                epoch_detected=metrics.latest_epoch.epoch if metrics.latest_epoch else None,
            ))

        return issues
