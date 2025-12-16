"""
Training Advisor module for Hunter Drone Detection System.

Provides training analysis, issue detection, and recommendations
for YOLO detector and Siamese embedder training.

Main components:
- Collectors: Parse training logs from various sources
- Analyzers: Detect training issues using rule-based logic
- Reporters: Generate reports in various formats
- Engine: Coordinate analysis and recommendations
- Auto-tuner: Automatically apply safe fixes

Example usage:
    from hunter.training_advisor import TrainingAdvisor

    advisor = TrainingAdvisor()
    report = advisor.analyze("path/to/training/logs")
    print(report)
"""

from .interfaces import IAnalyzer, ICollector, IReporter, IAction, ILLMProvider
from .domain import (
    TrainingMetrics,
    EpochMetrics,
    Issue,
    IssueType,
    IssueSeverity,
    Recommendation,
    RecommendationType,
)
from .config import AdvisorConfig
from .advisor import TrainingAdvisor

__all__ = [
    # Main class
    "TrainingAdvisor",
    "AdvisorConfig",
    # Interfaces
    "IAnalyzer",
    "ICollector",
    "IReporter",
    "IAction",
    "ILLMProvider",
    # Domain
    "TrainingMetrics",
    "EpochMetrics",
    "Issue",
    "IssueType",
    "IssueSeverity",
    "Recommendation",
    "RecommendationType",
]
