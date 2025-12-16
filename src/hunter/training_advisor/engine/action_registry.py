"""
Action registry for training advisor.

Manages and dispatches automated actions.
"""

from typing import Dict, List, Optional

from ..interfaces import IAction
from ..domain.recommendations import Recommendation, RecommendationType


class ActionRegistry:
    """
    Registry for automated training actions.

    Manages available actions and matches them to recommendations.
    """

    def __init__(self):
        """Initialize action registry."""
        self._actions: Dict[str, IAction] = {}
        self._type_mapping: Dict[RecommendationType, List[str]] = {}

    def register(self, action: IAction) -> None:
        """
        Register an action.

        Args:
            action: Action to register
        """
        self._actions[action.action_name] = action

    def unregister(self, action_name: str) -> None:
        """
        Unregister an action.

        Args:
            action_name: Name of action to remove
        """
        if action_name in self._actions:
            del self._actions[action_name]

    def list_actions(self) -> List[str]:
        """
        List all registered action names.

        Returns:
            List of action names
        """
        return list(self._actions.keys())

    def get_action(self, name: str) -> Optional[IAction]:
        """
        Get action by name.

        Args:
            name: Action name

        Returns:
            Action instance or None
        """
        return self._actions.get(name)

    def get_actions_for(self, recommendation: Recommendation) -> List[IAction]:
        """
        Get all actions that can handle a recommendation.

        Args:
            recommendation: Recommendation to find actions for

        Returns:
            List of applicable actions
        """
        applicable = []

        for action in self._actions.values():
            if action.can_apply(recommendation):
                applicable.append(action)

        return applicable

    def get_safe_actions(self) -> List[IAction]:
        """
        Get all safe actions.

        Returns:
            List of actions marked as safe
        """
        return [a for a in self._actions.values() if a.is_safe]

    def execute_action(
        self,
        action_name: str,
        recommendation: Recommendation,
        config_path: str,
    ) -> bool:
        """
        Execute a specific action.

        Args:
            action_name: Name of action to execute
            recommendation: Recommendation to apply
            config_path: Path to config file

        Returns:
            True if successful

        Raises:
            ValueError: If action not found
        """
        action = self._actions.get(action_name)
        if action is None:
            raise ValueError(f"Action not found: {action_name}")

        return action.execute(recommendation, config_path)
