"""
Domain module for training advisor.

Contains data classes for metrics, issues, and recommendations.
"""

from .metrics import TrainingMetrics, EpochMetrics
from .issues import Issue, IssueType, IssueSeverity
from .recommendations import Recommendation, RecommendationType

__all__ = [
    "TrainingMetrics",
    "EpochMetrics",
    "Issue",
    "IssueType",
    "IssueSeverity",
    "Recommendation",
    "RecommendationType",
]
