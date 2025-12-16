"""
Base collector class.

Abstract base class for all metrics collectors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from ..domain.metrics import TrainingMetrics, EpochMetrics


class BaseCollector(ABC):
    """
    Abstract base class for training metrics collectors.

    Subclasses must implement source_type property and _parse_source method.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source type this collector handles."""
        ...

    def collect(self, source: Union[Path, str]) -> TrainingMetrics:
        """
        Collect training metrics from source.

        Args:
            source: Path to log file or directory

        Returns:
            Parsed training metrics

        Raises:
            FileNotFoundError: If source doesn't exist
        """
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        epochs = self._parse_source(source)
        return TrainingMetrics(epochs=epochs)

    @abstractmethod
    def _parse_source(self, source: Path) -> List[EpochMetrics]:
        """
        Parse source and extract epoch metrics.

        Args:
            source: Path to source file or directory

        Returns:
            List of epoch metrics
        """
        ...
