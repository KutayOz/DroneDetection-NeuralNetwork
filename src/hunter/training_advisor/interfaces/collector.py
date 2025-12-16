"""
Collector interface.

Protocol class for training metrics collectors.
"""

from pathlib import Path
from typing import Protocol, Union, runtime_checkable

from ..domain.metrics import TrainingMetrics


@runtime_checkable
class ICollector(Protocol):
    """
    Protocol for training metrics collectors.

    Collectors parse training logs/outputs and extract metrics.
    Different collectors handle different log formats (YOLO, PyTorch, CSV, etc.).
    """

    @property
    def source_type(self) -> str:
        """Source type this collector handles (e.g., 'yolo', 'csv', 'tensorboard')."""
        ...

    def collect(self, source: Union[Path, str]) -> TrainingMetrics:
        """
        Collect training metrics from source.

        Args:
            source: Path to log file, directory, or other source

        Returns:
            Parsed training metrics
        """
        ...
