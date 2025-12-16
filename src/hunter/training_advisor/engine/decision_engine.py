"""
Decision engine for training advisor.

Coordinates analysis and generates recommendations.
"""

from typing import Dict, List, Any, Optional

from ..analyzers import get_all_analyzers, BaseAnalyzer
from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue
from ..domain.recommendations import Recommendation
from .recommendation_engine import RecommendationEngine


class DecisionEngine:
    """
    Main decision engine for training analysis.

    Coordinates analyzers, generates recommendations, and orchestrates
    the analysis pipeline.
    """

    def __init__(
        self,
        analyzers: Optional[List[BaseAnalyzer]] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
    ):
        """
        Initialize decision engine.

        Args:
            analyzers: List of analyzers to use (defaults to all)
            recommendation_engine: Recommendation engine (defaults to new instance)
        """
        self._analyzers = analyzers or get_all_analyzers()
        self._recommendation_engine = recommendation_engine or RecommendationEngine()

    def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Run all analyzers on metrics.

        Args:
            metrics: Training metrics to analyze

        Returns:
            List of detected issues
        """
        all_issues = []

        for analyzer in self._analyzers:
            try:
                issues = analyzer.analyze(metrics)
                all_issues.extend(issues)
            except Exception as e:
                # Log error but continue with other analyzers
                pass

        # Sort by severity (most severe first)
        all_issues.sort(key=lambda i: i.severity.value, reverse=True)

        return all_issues

    def recommend(self, issues: List[Issue]) -> List[Recommendation]:
        """
        Generate recommendations for issues.

        Args:
            issues: Detected issues

        Returns:
            List of recommendations
        """
        return self._recommendation_engine.generate(issues)

    def run_analysis(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """
        Run full analysis pipeline.

        Args:
            metrics: Training metrics

        Returns:
            Dict with issues, recommendations, and summary
        """
        issues = self.analyze(metrics)
        recommendations = self.recommend(issues)

        return {
            "issues": issues,
            "recommendations": recommendations,
            "summary": {
                "total_issues": len(issues),
                "total_recommendations": len(recommendations),
                "critical_issues": sum(1 for i in issues if i.severity.name == "CRITICAL"),
                "auto_applicable": sum(1 for r in recommendations if r.auto_applicable),
            },
        }
