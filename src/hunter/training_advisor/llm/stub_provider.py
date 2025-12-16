"""
Stub LLM provider for testing.

Returns pre-configured responses for testing purposes.
"""

from typing import List, Optional

from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue


class StubLLMProvider:
    """
    Stub LLM provider for testing.

    Returns configurable responses without actual LLM calls.
    """

    def __init__(self, response: str = "Stub LLM analysis"):
        """
        Initialize stub provider.

        Args:
            response: Response to return for all analyze calls
        """
        self._response = response

    @property
    def provider_name(self) -> str:
        """Provider name."""
        return "stub"

    def analyze(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
        context: Optional[str] = None,
    ) -> str:
        """
        Return stub analysis.

        Args:
            metrics: Training metrics (ignored)
            issues: Detected issues (ignored)
            context: Additional context (ignored)

        Returns:
            Pre-configured response
        """
        return self._response
