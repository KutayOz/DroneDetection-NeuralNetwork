"""
Auto-tuner for training advisor.

Automatically applies safe recommendations to training configurations.
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..domain.recommendations import Recommendation
from ..engine.action_registry import ActionRegistry
from ..actions import LearningRateAction, RegularizationAction


class AutoTuner:
    """
    Automatic configuration tuner.

    Applies safe recommendations automatically with backup support.
    """

    def __init__(
        self,
        action_registry: Optional[ActionRegistry] = None,
        backup_dir: str = "./config_backups",
        safe_only: bool = True,
    ):
        """
        Initialize auto-tuner.

        Args:
            action_registry: Registry of available actions
            backup_dir: Directory for config backups
            safe_only: Only apply safe actions
        """
        self._registry = action_registry or self._create_default_registry()
        self._backup_dir = Path(backup_dir)
        self._safe_only = safe_only

    def _create_default_registry(self) -> ActionRegistry:
        """Create registry with default actions."""
        registry = ActionRegistry()
        registry.register(LearningRateAction())
        registry.register(RegularizationAction())
        return registry

    def tune(
        self,
        recommendations: List[Recommendation],
        config_path: str,
        create_backup: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply recommendations to configuration.

        Args:
            recommendations: Recommendations to apply
            config_path: Path to training config file
            create_backup: Whether to backup before changes
            dry_run: Preview changes without applying

        Returns:
            Dict with results of tuning
        """
        results = {
            "applied": [],
            "skipped": [],
            "failed": [],
            "backup_path": None,
        }

        # Filter to auto-applicable recommendations
        applicable = [r for r in recommendations if r.auto_applicable]

        if not applicable:
            return results

        # Create backup
        if create_backup and not dry_run:
            results["backup_path"] = self._create_backup(config_path)

        # Apply each recommendation
        for rec in applicable:
            if dry_run:
                results["applied"].append({
                    "recommendation": rec.rec_type.name,
                    "dry_run": True,
                })
                continue

            actions = self._registry.get_actions_for(rec)

            if self._safe_only:
                actions = [a for a in actions if a.is_safe]

            if not actions:
                results["skipped"].append({
                    "recommendation": rec.rec_type.name,
                    "reason": "No applicable actions",
                })
                continue

            # Execute first applicable action
            action = actions[0]
            try:
                success = action.execute(rec, config_path)
                if success:
                    results["applied"].append({
                        "recommendation": rec.rec_type.name,
                        "action": action.action_name,
                    })
                else:
                    results["failed"].append({
                        "recommendation": rec.rec_type.name,
                        "reason": "Action returned False",
                    })
            except Exception as e:
                results["failed"].append({
                    "recommendation": rec.rec_type.name,
                    "reason": str(e),
                })

        return results

    def _create_backup(self, config_path: str) -> Optional[str]:
        """
        Create backup of configuration file.

        Args:
            config_path: Path to config file

        Returns:
            Path to backup file or None
        """
        source = Path(config_path)
        if not source.exists():
            return None

        self._backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_{timestamp}{source.suffix}"
        backup_path = self._backup_dir / backup_name

        shutil.copy2(source, backup_path)
        return str(backup_path)

    def restore_backup(self, backup_path: str, config_path: str) -> bool:
        """
        Restore configuration from backup.

        Args:
            backup_path: Path to backup file
            config_path: Path to restore to

        Returns:
            True if successful
        """
        backup = Path(backup_path)
        if not backup.exists():
            return False

        shutil.copy2(backup, config_path)
        return True
