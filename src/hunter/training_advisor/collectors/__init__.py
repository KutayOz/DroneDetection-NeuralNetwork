"""
Collectors module for training advisor.

Contains collectors for parsing training metrics from various sources.
"""

from .base import BaseCollector
from .yolo import YOLOCollector
from .csv_collector import CSVCollector
from .stub import StubCollector


def get_collector(source_type: str, **kwargs) -> BaseCollector:
    """
    Factory function to get collector by source type.

    Args:
        source_type: Type of collector ('yolo', 'csv', 'stub')
        **kwargs: Additional arguments for collector

    Returns:
        Collector instance

    Raises:
        ValueError: If unknown source type
    """
    collectors = {
        "yolo": YOLOCollector,
        "csv": CSVCollector,
        "stub": StubCollector,
    }

    if source_type not in collectors:
        raise ValueError(f"Unknown collector type: {source_type}. Available: {list(collectors.keys())}")

    return collectors[source_type](**kwargs)


__all__ = [
    "BaseCollector",
    "YOLOCollector",
    "CSVCollector",
    "StubCollector",
    "get_collector",
]
