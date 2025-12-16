"""
Unit tests for training_advisor config module.

Tests for Pydantic configuration classes.
"""

import pytest
from pathlib import Path
from typing import Optional


# ============================================
# AdvisorConfig Tests
# ============================================


class TestAdvisorConfig:
    """Tests for AdvisorConfig main configuration."""

    def test_creation_defaults(self):
        """AdvisorConfig can be created with defaults."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert config is not None

    def test_has_analyzer_config(self):
        """AdvisorConfig has analyzer configuration."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert hasattr(config, 'analyzers')

    def test_has_collector_config(self):
        """AdvisorConfig has collector configuration."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert hasattr(config, 'collectors')

    def test_has_reporter_config(self):
        """AdvisorConfig has reporter configuration."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert hasattr(config, 'reporters')

    def test_has_llm_config(self):
        """AdvisorConfig has LLM configuration."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert hasattr(config, 'llm')

    def test_has_auto_tune_config(self):
        """AdvisorConfig has auto-tune configuration."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        assert hasattr(config, 'auto_tune')

    def test_from_yaml(self, tmp_path):
        """AdvisorConfig can be loaded from YAML."""
        from hunter.training_advisor.config import AdvisorConfig

        yaml_content = """
analyzers:
  enabled:
    - overfitting
    - learning_rate
collectors:
  default_source: yolo
reporters:
  default_format: console
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = AdvisorConfig.from_yaml(yaml_file)
        assert config is not None

    def test_to_yaml(self, tmp_path):
        """AdvisorConfig can be saved to YAML."""
        from hunter.training_advisor.config import AdvisorConfig

        config = AdvisorConfig()
        yaml_file = tmp_path / "output.yaml"

        config.to_yaml(yaml_file)
        assert yaml_file.exists()


# ============================================
# AnalyzerConfig Tests
# ============================================


class TestAnalyzerConfig:
    """Tests for AnalyzerConfig."""

    def test_creation(self):
        """AnalyzerConfig can be created."""
        from hunter.training_advisor.config import AnalyzerConfig

        config = AnalyzerConfig()
        assert config is not None

    def test_enabled_analyzers(self):
        """AnalyzerConfig has enabled analyzers list."""
        from hunter.training_advisor.config import AnalyzerConfig

        config = AnalyzerConfig(enabled=["overfitting", "learning_rate"])
        assert "overfitting" in config.enabled
        assert "learning_rate" in config.enabled

    def test_default_all_enabled(self):
        """AnalyzerConfig enables all by default."""
        from hunter.training_advisor.config import AnalyzerConfig

        config = AnalyzerConfig()
        # Should have at least some default analyzers enabled
        assert len(config.enabled) > 0 or config.enable_all


# ============================================
# AnalysisThresholds Tests
# ============================================


class TestAnalysisThresholds:
    """Tests for analysis threshold configuration."""

    def test_creation(self):
        """AnalysisThresholds can be created."""
        from hunter.training_advisor.config import AnalysisThresholds

        thresholds = AnalysisThresholds()
        assert thresholds is not None

    def test_overfitting_thresholds(self):
        """AnalysisThresholds has overfitting thresholds."""
        from hunter.training_advisor.config import AnalysisThresholds

        thresholds = AnalysisThresholds()
        assert hasattr(thresholds, 'overfitting_gap_threshold')
        assert thresholds.overfitting_gap_threshold > 0

    def test_plateau_thresholds(self):
        """AnalysisThresholds has plateau detection thresholds."""
        from hunter.training_advisor.config import AnalysisThresholds

        thresholds = AnalysisThresholds()
        assert hasattr(thresholds, 'plateau_epochs')
        assert hasattr(thresholds, 'plateau_min_improvement')

    def test_lr_thresholds(self):
        """AnalysisThresholds has learning rate thresholds."""
        from hunter.training_advisor.config import AnalysisThresholds

        thresholds = AnalysisThresholds()
        assert hasattr(thresholds, 'lr_divergence_threshold')

    def test_map_thresholds(self):
        """AnalysisThresholds has mAP thresholds."""
        from hunter.training_advisor.config import AnalysisThresholds

        thresholds = AnalysisThresholds()
        assert hasattr(thresholds, 'low_map_threshold')


# ============================================
# CollectorConfig Tests
# ============================================


class TestCollectorConfig:
    """Tests for CollectorConfig."""

    def test_creation(self):
        """CollectorConfig can be created."""
        from hunter.training_advisor.config import CollectorConfig

        config = CollectorConfig()
        assert config is not None

    def test_default_source(self):
        """CollectorConfig has default source type."""
        from hunter.training_advisor.config import CollectorConfig

        config = CollectorConfig(default_source="yolo")
        assert config.default_source == "yolo"

    def test_yolo_results_path(self):
        """CollectorConfig has YOLO results path pattern."""
        from hunter.training_advisor.config import CollectorConfig

        config = CollectorConfig()
        assert hasattr(config, 'yolo_results_pattern')


# ============================================
# ReporterConfig Tests
# ============================================


class TestReporterConfig:
    """Tests for ReporterConfig."""

    def test_creation(self):
        """ReporterConfig can be created."""
        from hunter.training_advisor.config import ReporterConfig

        config = ReporterConfig()
        assert config is not None

    def test_default_format(self):
        """ReporterConfig has default format."""
        from hunter.training_advisor.config import ReporterConfig

        config = ReporterConfig(default_format="markdown")
        assert config.default_format == "markdown"

    def test_output_path(self):
        """ReporterConfig has output path."""
        from hunter.training_advisor.config import ReporterConfig

        config = ReporterConfig(output_dir="./reports")
        assert config.output_dir == "./reports"


# ============================================
# LLMConfig Tests
# ============================================


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_creation(self):
        """LLMConfig can be created."""
        from hunter.training_advisor.config import LLMConfig

        config = LLMConfig()
        assert config is not None

    def test_enabled_default(self):
        """LLMConfig disabled by default."""
        from hunter.training_advisor.config import LLMConfig

        config = LLMConfig()
        assert config.enabled is False

    def test_provider_setting(self):
        """LLMConfig has provider setting."""
        from hunter.training_advisor.config import LLMConfig

        config = LLMConfig(enabled=True, provider="anthropic")
        assert config.provider == "anthropic"

    def test_model_setting(self):
        """LLMConfig has model setting."""
        from hunter.training_advisor.config import LLMConfig

        config = LLMConfig(enabled=True, model="claude-3-sonnet-20240229")
        assert config.model == "claude-3-sonnet-20240229"


# ============================================
# AutoTuneConfig Tests
# ============================================


class TestAutoTuneConfig:
    """Tests for AutoTuneConfig."""

    def test_creation(self):
        """AutoTuneConfig can be created."""
        from hunter.training_advisor.config import AutoTuneConfig

        config = AutoTuneConfig()
        assert config is not None

    def test_enabled_default(self):
        """AutoTuneConfig disabled by default."""
        from hunter.training_advisor.config import AutoTuneConfig

        config = AutoTuneConfig()
        assert config.enabled is False

    def test_safe_only_default(self):
        """AutoTuneConfig applies only safe actions by default."""
        from hunter.training_advisor.config import AutoTuneConfig

        config = AutoTuneConfig(enabled=True)
        assert config.safe_only is True

    def test_backup_enabled(self):
        """AutoTuneConfig creates backups by default."""
        from hunter.training_advisor.config import AutoTuneConfig

        config = AutoTuneConfig(enabled=True)
        assert config.create_backup is True


# ============================================
# Config Module Exports Tests
# ============================================


class TestConfigExports:
    """Tests for config module exports."""

    def test_exports_advisor_config(self):
        """AdvisorConfig is exported."""
        from hunter.training_advisor.config import AdvisorConfig
        assert AdvisorConfig is not None

    def test_exports_analyzer_config(self):
        """AnalyzerConfig is exported."""
        from hunter.training_advisor.config import AnalyzerConfig
        assert AnalyzerConfig is not None

    def test_exports_analysis_thresholds(self):
        """AnalysisThresholds is exported."""
        from hunter.training_advisor.config import AnalysisThresholds
        assert AnalysisThresholds is not None

    def test_exports_collector_config(self):
        """CollectorConfig is exported."""
        from hunter.training_advisor.config import CollectorConfig
        assert CollectorConfig is not None

    def test_exports_reporter_config(self):
        """ReporterConfig is exported."""
        from hunter.training_advisor.config import ReporterConfig
        assert ReporterConfig is not None

    def test_exports_llm_config(self):
        """LLMConfig is exported."""
        from hunter.training_advisor.config import LLMConfig
        assert LLMConfig is not None

    def test_exports_auto_tune_config(self):
        """AutoTuneConfig is exported."""
        from hunter.training_advisor.config import AutoTuneConfig
        assert AutoTuneConfig is not None
