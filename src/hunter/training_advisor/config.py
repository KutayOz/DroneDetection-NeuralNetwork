"""
Configuration for Training Advisor.

Pydantic-based configuration classes for all components.
"""

from pathlib import Path
from typing import List, Optional, Union, Any
from dataclasses import dataclass, field

import yaml


@dataclass
class AnalysisThresholds:
    """Thresholds for training analysis."""

    # Overfitting detection
    overfitting_gap_threshold: float = 0.1  # val_loss - train_loss gap
    overfitting_epochs: int = 3  # consecutive epochs of increasing gap

    # Underfitting detection
    underfitting_loss_threshold: float = 0.5  # high loss after many epochs
    underfitting_min_epochs: int = 10

    # Learning rate issues
    lr_divergence_threshold: float = 2.0  # loss increase ratio
    lr_oscillation_threshold: float = 0.2  # loss variance
    lr_too_slow_epochs: int = 10  # epochs with minimal progress

    # Plateau detection
    plateau_epochs: int = 5  # consecutive epochs with no improvement
    plateau_min_improvement: float = 0.001  # minimum loss decrease

    # Detection metrics (YOLO)
    low_map_threshold: float = 0.3  # minimum acceptable mAP@50
    precision_recall_imbalance: float = 0.2  # |precision - recall| threshold

    # Embedding metrics (Siamese)
    embedding_convergence_threshold: float = 0.01  # minimum improvement
    triplet_stuck_threshold: float = 0.001  # margin of triplet loss change


@dataclass
class AnalyzerConfig:
    """Configuration for analyzers."""

    enabled: List[str] = field(default_factory=lambda: [
        "overfitting",
        "learning_rate",
        "convergence",
        "detection",
        "siamese",
    ])
    enable_all: bool = True
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)


@dataclass
class CollectorConfig:
    """Configuration for metrics collectors."""

    default_source: str = "yolo"
    yolo_results_pattern: str = "results.csv"
    tensorboard_log_dir: str = "runs"
    csv_delimiter: str = ","
    auto_detect: bool = True


@dataclass
class ReporterConfig:
    """Configuration for report generators."""

    default_format: str = "console"
    output_dir: str = "./training_reports"
    include_timestamps: bool = True
    include_recommendations: bool = True
    include_details: bool = True
    markdown_style: str = "github"


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""

    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-3-sonnet-20240229"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 2000
    temperature: float = 0.3
    timeout_s: float = 30.0


@dataclass
class AutoTuneConfig:
    """Configuration for auto-tuning."""

    enabled: bool = False
    safe_only: bool = True  # Only apply safe actions
    create_backup: bool = True  # Backup config before changes
    backup_dir: str = "./config_backups"
    dry_run: bool = False  # Preview changes without applying
    require_confirmation: bool = True  # Ask before applying


@dataclass
class AdvisorConfig:
    """Main configuration for Training Advisor."""

    analyzers: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    collectors: CollectorConfig = field(default_factory=CollectorConfig)
    reporters: ReporterConfig = field(default_factory=ReporterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    auto_tune: AutoTuneConfig = field(default_factory=AutoTuneConfig)

    # General settings
    verbose: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AdvisorConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AdvisorConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "AdvisorConfig":
        """Create config from dictionary."""
        config = cls()

        if "analyzers" in data:
            analyzer_data = data["analyzers"]
            if "enabled" in analyzer_data:
                config.analyzers.enabled = analyzer_data["enabled"]
            if "enable_all" in analyzer_data:
                config.analyzers.enable_all = analyzer_data["enable_all"]
            if "thresholds" in analyzer_data:
                for key, value in analyzer_data["thresholds"].items():
                    if hasattr(config.analyzers.thresholds, key):
                        setattr(config.analyzers.thresholds, key, value)

        if "collectors" in data:
            for key, value in data["collectors"].items():
                if hasattr(config.collectors, key):
                    setattr(config.collectors, key, value)

        if "reporters" in data:
            for key, value in data["reporters"].items():
                if hasattr(config.reporters, key):
                    setattr(config.reporters, key, value)

        if "llm" in data:
            for key, value in data["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        if "auto_tune" in data:
            for key, value in data["auto_tune"].items():
                if hasattr(config.auto_tune, key):
                    setattr(config.auto_tune, key, value)

        if "verbose" in data:
            config.verbose = data["verbose"]
        if "log_level" in data:
            config.log_level = data["log_level"]

        return config

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()

        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "analyzers": {
                "enabled": self.analyzers.enabled,
                "enable_all": self.analyzers.enable_all,
                "thresholds": {
                    "overfitting_gap_threshold": self.analyzers.thresholds.overfitting_gap_threshold,
                    "overfitting_epochs": self.analyzers.thresholds.overfitting_epochs,
                    "plateau_epochs": self.analyzers.thresholds.plateau_epochs,
                    "plateau_min_improvement": self.analyzers.thresholds.plateau_min_improvement,
                    "lr_divergence_threshold": self.analyzers.thresholds.lr_divergence_threshold,
                    "low_map_threshold": self.analyzers.thresholds.low_map_threshold,
                },
            },
            "collectors": {
                "default_source": self.collectors.default_source,
                "yolo_results_pattern": self.collectors.yolo_results_pattern,
                "auto_detect": self.collectors.auto_detect,
            },
            "reporters": {
                "default_format": self.reporters.default_format,
                "output_dir": self.reporters.output_dir,
                "include_timestamps": self.reporters.include_timestamps,
            },
            "llm": {
                "enabled": self.llm.enabled,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
            "auto_tune": {
                "enabled": self.auto_tune.enabled,
                "safe_only": self.auto_tune.safe_only,
                "create_backup": self.auto_tune.create_backup,
            },
            "verbose": self.verbose,
            "log_level": self.log_level,
        }


__all__ = [
    "AdvisorConfig",
    "AnalyzerConfig",
    "AnalysisThresholds",
    "CollectorConfig",
    "ReporterConfig",
    "LLMConfig",
    "AutoTuneConfig",
]
