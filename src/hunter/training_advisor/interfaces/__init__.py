"""
Interfaces module for training advisor.

Contains Protocol classes defining contracts for analyzers, collectors, and reporters.
"""

from .analyzer import IAnalyzer, IAction, ILLMProvider
from .collector import ICollector
from .reporter import IReporter

__all__ = [
    "IAnalyzer",
    "IAction",
    "ILLMProvider",
    "ICollector",
    "IReporter",
]
