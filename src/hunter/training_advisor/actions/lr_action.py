"""
Learning rate action.

Action for modifying learning rate configuration.
"""

import yaml
from pathlib import Path

from ..domain.recommendations import Recommendation, RecommendationType


class LearningRateAction:
    """
    Action for adjusting learning rate.

    Handles REDUCE_LR and INCREASE_LR recommendations.
    """

    def __init__(self, reduction_factor: float = 0.5, increase_factor: float = 2.0):
        """
        Initialize LR action.

        Args:
            reduction_factor: Factor to reduce LR by
            increase_factor: Factor to increase LR by
        """
        self._reduction_factor = reduction_factor
        self._increase_factor = increase_factor

    @property
    def action_name(self) -> str:
        """Action name."""
        return "learning_rate"

    @property
    def is_safe(self) -> bool:
        """Whether action is safe to auto-apply."""
        return True

    def can_apply(self, recommendation: Recommendation) -> bool:
        """Check if action can handle recommendation."""
        return recommendation.rec_type in (
            RecommendationType.REDUCE_LR,
            RecommendationType.INCREASE_LR,
        )

    def execute(self, recommendation: Recommendation, config_path: str) -> bool:
        """
        Execute LR modification.

        Args:
            recommendation: Recommendation to apply
            config_path: Path to config file

        Returns:
            True if successful
        """
        path = Path(config_path)
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}

            current_lr = config.get("lr0", 0.01)

            if recommendation.rec_type == RecommendationType.REDUCE_LR:
                new_lr = current_lr * self._reduction_factor
            else:
                new_lr = current_lr * self._increase_factor

            config["lr0"] = new_lr

            with open(path, "w") as f:
                yaml.safe_dump(config, f)

            return True

        except Exception:
            return False
