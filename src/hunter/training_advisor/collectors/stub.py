"""
Stub collector for testing.

Returns pre-configured or generated metrics for testing purposes.
"""

from pathlib import Path
from typing import List, Optional, Union

from .base import BaseCollector
from ..domain.metrics import TrainingMetrics, EpochMetrics


class StubCollector(BaseCollector):
    """
    Stub collector for testing.

    Returns pre-configured metrics or generates synthetic data.
    """

    def __init__(
        self,
        metrics: Optional[TrainingMetrics] = None,
        num_epochs: int = 10,
        initial_loss: float = 1.0,
        loss_decay: float = 0.9,
    ):
        """
        Initialize stub collector.

        Args:
            metrics: Pre-configured metrics to return
            num_epochs: Number of epochs if generating metrics
            initial_loss: Initial loss value for generation
            loss_decay: Loss decay factor per epoch
        """
        self._metrics = metrics
        self._num_epochs = num_epochs
        self._initial_loss = initial_loss
        self._loss_decay = loss_decay

    @property
    def source_type(self) -> str:
        """Source type identifier."""
        return "stub"

    def collect(self, source: Union[Path, str]) -> TrainingMetrics:
        """
        Return stub metrics (ignores source).

        Args:
            source: Ignored - stub doesn't need actual source

        Returns:
            Pre-configured or generated metrics
        """
        if self._metrics is not None:
            return self._metrics

        return TrainingMetrics(epochs=self._generate_epochs())

    def _parse_source(self, source: Path) -> List[EpochMetrics]:
        """Not used for stub collector."""
        return self._generate_epochs()

    def _generate_epochs(self) -> List[EpochMetrics]:
        """
        Generate synthetic epoch metrics.

        Returns:
            List of generated epoch metrics
        """
        epochs = []
        train_loss = self._initial_loss
        val_loss = self._initial_loss * 1.1  # Val slightly higher

        for i in range(self._num_epochs):
            epochs.append(
                EpochMetrics(
                    epoch=i,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.01 * (self._loss_decay ** i),
                )
            )
            train_loss *= self._loss_decay
            val_loss *= self._loss_decay

        return epochs
