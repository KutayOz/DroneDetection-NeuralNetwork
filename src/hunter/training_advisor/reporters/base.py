"""
Base reporter class.

Abstract base class for all report generators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..domain.issues import Issue
from ..domain.recommendations import Recommendation


class BaseReporter(ABC):
    """
    Abstract base class for report generators.

    Subclasses must implement format property and report method.
    """

    @property
    @abstractmethod
    def format(self) -> str:
        """Report format identifier."""
        ...

    @abstractmethod
    def report(
        self,
        issues: List[Issue],
        recommendations: List[Recommendation],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate report.

        Args:
            issues: Detected issues
            recommendations: Generated recommendations
            output_path: Optional path to write report

        Returns:
            Report content as string
        """
        ...
