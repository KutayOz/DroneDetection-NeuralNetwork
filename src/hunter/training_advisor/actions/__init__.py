"""
Actions module for training advisor.

Contains automated actions that can modify training configurations.
"""

from .lr_action import LearningRateAction
from .regularization_action import RegularizationAction

__all__ = [
    "LearningRateAction",
    "RegularizationAction",
]
