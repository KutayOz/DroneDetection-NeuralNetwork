"""
Analyzer interfaces.

Protocol classes for analyzers, actions, and LLM providers.
"""

from typing import List, Optional, Protocol, runtime_checkable

from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue
from ..domain.recommendations import Recommendation


@runtime_checkable
class IAnalyzer(Protocol):
    """
    Protocol for training analyzers.

    Analyzers examine training metrics and detect issues.
    Each analyzer should focus on detecting specific types of issues.
    """

    @property
    def name(self) -> str:
        """Analyzer name for identification."""
        ...

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Analyze training metrics and detect issues.

        Args:
            metrics: Training metrics to analyze

        Returns:
            List of detected issues
        """
        ...


@runtime_checkable
class IAction(Protocol):
    """
    Protocol for automated actions.

    Actions can auto-apply recommendations to training configurations.
    """

    @property
    def action_name(self) -> str:
        """Action name for identification."""
        ...

    @property
    def is_safe(self) -> bool:
        """Whether this action is safe to auto-apply."""
        ...

    def can_apply(self, recommendation: Recommendation) -> bool:
        """
        Check if this action can apply a recommendation.

        Args:
            recommendation: Recommendation to check

        Returns:
            True if action can handle this recommendation
        """
        ...

    def execute(self, recommendation: Recommendation, config_path: str) -> bool:
        """
        Execute the action to apply recommendation.

        Args:
            recommendation: Recommendation to apply
            config_path: Path to configuration file

        Returns:
            True if successfully applied
        """
        ...


@runtime_checkable
class ILLMProvider(Protocol):
    """
    Protocol for LLM analysis providers.

    LLM providers offer deep analysis of training issues using
    large language models.
    """

    @property
    def provider_name(self) -> str:
        """Provider name for identification."""
        ...

    def analyze(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
        context: Optional[str] = None,
    ) -> str:
        """
        Perform LLM analysis on training metrics and issues.

        Args:
            metrics: Training metrics
            issues: Detected issues
            context: Optional additional context

        Returns:
            LLM analysis as string
        """
        ...
