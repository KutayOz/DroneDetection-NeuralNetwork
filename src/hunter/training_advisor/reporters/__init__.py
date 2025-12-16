"""
Reporters module for training advisor.

Contains report generators for various output formats.
"""

from .base import BaseReporter
from .console_reporter import ConsoleReporter
from .markdown_reporter import MarkdownReporter


def get_reporter(format: str, **kwargs) -> BaseReporter:
    """
    Factory function to get reporter by format.

    Args:
        format: Report format ('console', 'markdown')
        **kwargs: Additional arguments for reporter

    Returns:
        Reporter instance

    Raises:
        ValueError: If unknown format
    """
    reporters = {
        "console": ConsoleReporter,
        "markdown": MarkdownReporter,
    }

    if format not in reporters:
        raise ValueError(f"Unknown reporter format: {format}")

    return reporters[format](**kwargs)


__all__ = [
    "BaseReporter",
    "ConsoleReporter",
    "MarkdownReporter",
    "get_reporter",
]
