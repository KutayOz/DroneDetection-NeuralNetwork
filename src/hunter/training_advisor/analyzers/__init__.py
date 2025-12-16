"""
Analyzers module for training advisor.

Contains rule-based analyzers for detecting training issues.
"""

from typing import List

from .base import BaseAnalyzer
from .overfitting import OverfittingAnalyzer
from .learning_rate import LearningRateAnalyzer
from .convergence import ConvergenceAnalyzer
from .detection import DetectionAnalyzer
from .siamese import SiameseAnalyzer


def get_analyzer(name: str, **kwargs) -> BaseAnalyzer:
    """
    Factory function to get analyzer by name.

    Args:
        name: Analyzer name
        **kwargs: Additional arguments for analyzer

    Returns:
        Analyzer instance

    Raises:
        ValueError: If unknown analyzer name
    """
    analyzers = {
        "overfitting": OverfittingAnalyzer,
        "learning_rate": LearningRateAnalyzer,
        "convergence": ConvergenceAnalyzer,
        "detection": DetectionAnalyzer,
        "siamese": SiameseAnalyzer,
    }

    if name not in analyzers:
        raise ValueError(f"Unknown analyzer: {name}. Available: {list(analyzers.keys())}")

    return analyzers[name](**kwargs)


def get_all_analyzers(**kwargs) -> List[BaseAnalyzer]:
    """
    Get all available analyzers.

    Args:
        **kwargs: Shared arguments for analyzers

    Returns:
        List of all analyzer instances
    """
    return [
        OverfittingAnalyzer(**kwargs),
        LearningRateAnalyzer(**kwargs),
        ConvergenceAnalyzer(**kwargs),
        DetectionAnalyzer(**kwargs),
        SiameseAnalyzer(**kwargs),
    ]


__all__ = [
    "BaseAnalyzer",
    "OverfittingAnalyzer",
    "LearningRateAnalyzer",
    "ConvergenceAnalyzer",
    "DetectionAnalyzer",
    "SiameseAnalyzer",
    "get_analyzer",
    "get_all_analyzers",
]
