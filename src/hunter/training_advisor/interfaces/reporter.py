"""
Reporter interface.

Protocol class for report generators.
"""

from typing import List, Optional, Protocol, runtime_checkable

from ..domain.issues import Issue
from ..domain.recommendations import Recommendation


@runtime_checkable
class IReporter(Protocol):
    """
    Protocol for report generators.

    Reporters generate reports from issues and recommendations
    in various formats (console, markdown, HTML, JSON).
    """

    @property
    def format(self) -> str:
        """Report format (e.g., 'console', 'markdown', 'html', 'json')."""
        ...

    def report(
        self,
        issues: List[Issue],
        recommendations: List[Recommendation],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate report from issues and recommendations.

        Args:
            issues: Detected issues
            recommendations: Generated recommendations
            output_path: Optional path to write report

        Returns:
            Report content as string
        """
        ...
