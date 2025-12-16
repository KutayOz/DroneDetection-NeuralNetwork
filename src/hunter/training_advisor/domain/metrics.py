"""
Training metrics data classes.

Contains data structures for training epoch metrics and aggregated metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EpochMetrics:
    """
    Metrics for a single training epoch.

    Contains loss values, learning rate, and optional detection/embedding metrics.
    """

    epoch: int
    train_loss: float
    val_loss: float

    # Optional - learning rate
    learning_rate: Optional[float] = None

    # Optional - detection metrics (YOLO)
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Optional - component losses (YOLO)
    box_loss: Optional[float] = None
    cls_loss: Optional[float] = None
    dfl_loss: Optional[float] = None

    # Optional - embedding metrics (Siamese)
    embedding_loss: Optional[float] = None
    triplet_loss: Optional[float] = None

    # Optional - timing
    epoch_time_s: Optional[float] = None

    @property
    def loss_gap(self) -> float:
        """Calculate gap between validation and training loss."""
        return self.val_loss - self.train_loss

    def is_improving(self, previous: "EpochMetrics") -> bool:
        """
        Check if validation loss improved compared to previous epoch.

        Args:
            previous: Previous epoch metrics

        Returns:
            True if val_loss decreased
        """
        return self.val_loss < previous.val_loss


@dataclass
class TrainingMetrics:
    """
    Aggregated training metrics across all epochs.

    Contains list of epoch metrics and provides helper methods
    for analysis.
    """

    epochs: List[EpochMetrics] = field(default_factory=list)

    # Optional metadata
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    total_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    initial_lr: Optional[float] = None

    @property
    def latest_epoch(self) -> Optional[EpochMetrics]:
        """Get the most recent epoch metrics."""
        if not self.epochs:
            return None
        return self.epochs[-1]

    @property
    def best_epoch(self) -> Optional[EpochMetrics]:
        """Get the epoch with lowest validation loss."""
        if not self.epochs:
            return None
        return min(self.epochs, key=lambda e: e.val_loss)

    @property
    def has_detection_metrics(self) -> bool:
        """Check if any epoch has detection metrics (mAP)."""
        return any(e.map50 is not None or e.map50_95 is not None for e in self.epochs)

    @property
    def has_embedding_metrics(self) -> bool:
        """Check if any epoch has embedding metrics."""
        return any(
            e.embedding_loss is not None or e.triplet_loss is not None
            for e in self.epochs
        )

    @property
    def epoch_count(self) -> int:
        """Get number of epochs."""
        return len(self.epochs)

    def get_loss_trend(self, window: int = 5) -> List[float]:
        """
        Get validation loss trend over recent epochs.

        Args:
            window: Number of epochs to consider

        Returns:
            List of val_loss values
        """
        recent = self.epochs[-window:] if len(self.epochs) >= window else self.epochs
        return [e.val_loss for e in recent]

    def get_train_val_gap_trend(self, window: int = 5) -> List[float]:
        """
        Get train-val loss gap trend over recent epochs.

        Args:
            window: Number of epochs to consider

        Returns:
            List of gap values (val_loss - train_loss)
        """
        recent = self.epochs[-window:] if len(self.epochs) >= window else self.epochs
        return [e.loss_gap for e in recent]
