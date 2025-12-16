"""
Regularization action.

Action for modifying regularization configuration.
"""

import yaml
from pathlib import Path

from ..domain.recommendations import Recommendation, RecommendationType


class RegularizationAction:
    """
    Action for adjusting regularization.

    Handles ADD_REGULARIZATION and weight decay recommendations.
    """

    def __init__(self, default_weight_decay: float = 0.0005):
        """
        Initialize regularization action.

        Args:
            default_weight_decay: Default weight decay value
        """
        self._default_weight_decay = default_weight_decay

    @property
    def action_name(self) -> str:
        """Action name."""
        return "regularization"

    @property
    def is_safe(self) -> bool:
        """Whether action is safe to auto-apply."""
        return True

    def can_apply(self, recommendation: Recommendation) -> bool:
        """Check if action can handle recommendation."""
        return recommendation.rec_type in (
            RecommendationType.ADD_REGULARIZATION,
            RecommendationType.ADD_WEIGHT_DECAY,
        )

    def execute(self, recommendation: Recommendation, config_path: str) -> bool:
        """
        Execute regularization modification.

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

            value = recommendation.suggested_value or self._default_weight_decay
            config["weight_decay"] = value

            with open(path, "w") as f:
                yaml.safe_dump(config, f)

            return True

        except Exception:
            return False
