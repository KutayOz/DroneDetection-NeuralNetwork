"""
Base analyzer class.

Abstract base class for all training analyzers.
"""

from abc import ABC, abstractmethod
from typing import List

from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue


class BaseAnalyzer(ABC):
    """
    Abstract base class for training analyzers.

    Subclasses must implement name property and analyze method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Analyzer name for identification."""
        ...

    @abstractmethod
    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze training metrics and detect issues.

        Args:
            metrics: Training metrics to analyze

        Returns:
            List of detected issues
        """
        ...
