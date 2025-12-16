"""
Unit tests for main TrainingAdvisor class.

Integration tests for the complete training advisor system.
"""

import pytest
from pathlib import Path


# ============================================
# TrainingAdvisor Tests
# ============================================


class TestTrainingAdvisor:
    """Tests for TrainingAdvisor main class."""

    def test_class_exists(self):
        """TrainingAdvisor class exists."""
        from hunter.training_advisor import TrainingAdvisor
        assert TrainingAdvisor is not None

    def test_creation_default(self):
        """TrainingAdvisor can be created with defaults."""
        from hunter.training_advisor import TrainingAdvisor

        advisor = TrainingAdvisor()
        assert advisor is not None

    def test_has_config(self):
        """TrainingAdvisor has config property."""
        from hunter.training_advisor import TrainingAdvisor

        advisor = TrainingAdvisor()
        assert advisor.config is not None

    def test_collect_method(self, tmp_path):
        """TrainingAdvisor.collect returns metrics."""
        from hunter.training_advisor import TrainingAdvisor, TrainingMetrics

        # Create test CSV
        csv_file = tmp_path / "training.csv"
        csv_file.write_text("epoch,train_loss,val_loss\n0,0.5,0.6\n1,0.4,0.5\n")

        advisor = TrainingAdvisor()
        metrics = advisor.collect(csv_file, source_type="csv")

        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.epochs) == 2

    def test_detect_issues_method(self):
        """TrainingAdvisor.detect_issues returns issues."""
        from hunter.training_advisor import TrainingAdvisor, TrainingMetrics, EpochMetrics

        advisor = TrainingAdvisor()

        # Create metrics with overfitting pattern
        epochs = [
            EpochMetrics(epoch=i, train_loss=1.0 - i * 0.1, val_loss=1.0 + i * 0.1)
            for i in range(10)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        issues = advisor.detect_issues(metrics)
        assert isinstance(issues, list)

    def test_recommend_method(self):
        """TrainingAdvisor.recommend returns recommendations."""
        from hunter.training_advisor import (
            TrainingAdvisor,
            Issue,
            IssueType,
            IssueSeverity,
        )

        advisor = TrainingAdvisor()
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.HIGH,
                message="Test",
                epoch_detected=5,
            )
        ]

        recommendations = advisor.recommend(issues)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_report_method(self):
        """TrainingAdvisor.generate_report returns report string."""
        from hunter.training_advisor import (
            TrainingAdvisor,
            Issue,
            IssueType,
            IssueSeverity,
            Recommendation,
            RecommendationType,
        )

        advisor = TrainingAdvisor()
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.HIGH,
                message="Test overfitting",
                epoch_detected=5,
            )
        ]
        recommendations = [
            Recommendation(
                rec_type=RecommendationType.EARLY_STOPPING,
                source_issue=IssueType.OVERFITTING,
                message="Enable early stopping",
            )
        ]

        report = advisor.generate_report(issues, recommendations)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_analyze_full_pipeline(self, tmp_path):
        """TrainingAdvisor.analyze runs full pipeline."""
        from hunter.training_advisor import TrainingAdvisor

        # Create test CSV with overfitting pattern
        csv_file = tmp_path / "training.csv"
        csv_content = "epoch,train_loss,val_loss\n"
        for i in range(15):
            train = 1.0 - i * 0.05
            val = 1.0 + i * 0.02
            csv_content += f"{i},{train},{val}\n"
        csv_file.write_text(csv_content)

        advisor = TrainingAdvisor()
        report = advisor.analyze(csv_file, source_type="csv")

        assert isinstance(report, str)
        assert len(report) > 0

    def test_run_full_analysis(self, tmp_path):
        """TrainingAdvisor.run_full_analysis returns complete results."""
        from hunter.training_advisor import TrainingAdvisor

        csv_file = tmp_path / "training.csv"
        csv_file.write_text("epoch,train_loss,val_loss\n0,0.5,0.6\n1,0.4,0.5\n")

        advisor = TrainingAdvisor()
        result = advisor.run_full_analysis(csv_file)

        assert "metrics" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "reports" in result


# ============================================
# Reporter Tests
# ============================================


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_class_exists(self):
        """ConsoleReporter class exists."""
        from hunter.training_advisor.reporters import ConsoleReporter
        assert ConsoleReporter is not None

    def test_format_property(self):
        """ConsoleReporter has format property."""
        from hunter.training_advisor.reporters import ConsoleReporter

        reporter = ConsoleReporter()
        assert reporter.format == "console"

    def test_report_generates_output(self):
        """ConsoleReporter generates report."""
        from hunter.training_advisor.reporters import ConsoleReporter
        from hunter.training_advisor import Issue, IssueType, IssueSeverity

        reporter = ConsoleReporter(use_colors=False)
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.HIGH,
                message="Test",
                epoch_detected=5,
            )
        ]

        report = reporter.report(issues, [])
        assert "Training Analysis Report" in report
        assert "OVERFITTING" in report


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def test_class_exists(self):
        """MarkdownReporter class exists."""
        from hunter.training_advisor.reporters import MarkdownReporter
        assert MarkdownReporter is not None

    def test_format_property(self):
        """MarkdownReporter has format property."""
        from hunter.training_advisor.reporters import MarkdownReporter

        reporter = MarkdownReporter()
        assert reporter.format == "markdown"

    def test_report_generates_markdown(self):
        """MarkdownReporter generates markdown."""
        from hunter.training_advisor.reporters import MarkdownReporter
        from hunter.training_advisor import Issue, IssueType, IssueSeverity

        reporter = MarkdownReporter()
        issues = [
            Issue(
                issue_type=IssueType.LR_TOO_HIGH,
                severity=IssueSeverity.CRITICAL,
                message="Test",
                epoch_detected=3,
            )
        ]

        report = reporter.report(issues, [])
        assert "# Training Analysis Report" in report
        assert "LR_TOO_HIGH" in report

    def test_writes_to_file(self, tmp_path):
        """MarkdownReporter writes to file."""
        from hunter.training_advisor.reporters import MarkdownReporter

        reporter = MarkdownReporter()
        output_file = tmp_path / "report.md"

        report = reporter.report([], [], output_path=str(output_file))

        assert output_file.exists()
        assert output_file.read_text() == report


# ============================================
# AutoTuner Tests
# ============================================


class TestAutoTuner:
    """Tests for AutoTuner."""

    def test_class_exists(self):
        """AutoTuner class exists."""
        from hunter.training_advisor.tuner import AutoTuner
        assert AutoTuner is not None

    def test_tune_dry_run(self, tmp_path):
        """AutoTuner dry run doesn't modify files."""
        from hunter.training_advisor.tuner import AutoTuner
        from hunter.training_advisor import Recommendation, RecommendationType, IssueType

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("lr0: 0.01\n")
        original_content = config_file.read_text()

        tuner = AutoTuner()
        recommendations = [
            Recommendation(
                rec_type=RecommendationType.REDUCE_LR,
                source_issue=IssueType.LR_TOO_HIGH,
                message="Reduce LR",
                auto_applicable=True,
            )
        ]

        result = tuner.tune(recommendations, str(config_file), dry_run=True)

        assert config_file.read_text() == original_content
        assert "applied" in result

    def test_creates_backup(self, tmp_path):
        """AutoTuner creates backup before changes."""
        from hunter.training_advisor.tuner import AutoTuner
        from hunter.training_advisor import Recommendation, RecommendationType, IssueType

        config_file = tmp_path / "config.yaml"
        config_file.write_text("lr0: 0.01\n")
        backup_dir = tmp_path / "backups"

        tuner = AutoTuner(backup_dir=str(backup_dir))
        recommendations = [
            Recommendation(
                rec_type=RecommendationType.REDUCE_LR,
                source_issue=IssueType.LR_TOO_HIGH,
                message="Reduce LR",
                auto_applicable=True,
            )
        ]

        result = tuner.tune(recommendations, str(config_file), create_backup=True, dry_run=False)

        assert backup_dir.exists()
        assert result.get("backup_path") is not None


# ============================================
# Module Exports Tests
# ============================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_training_advisor_exported(self):
        """TrainingAdvisor is exported from main module."""
        from hunter.training_advisor import TrainingAdvisor
        assert TrainingAdvisor is not None

    def test_advisor_config_exported(self):
        """AdvisorConfig is exported from main module."""
        from hunter.training_advisor import AdvisorConfig
        assert AdvisorConfig is not None

    def test_can_import_all_components(self):
        """All main components can be imported."""
        from hunter.training_advisor import (
            TrainingAdvisor,
            AdvisorConfig,
            TrainingMetrics,
            EpochMetrics,
            Issue,
            IssueType,
            IssueSeverity,
            Recommendation,
            RecommendationType,
            IAnalyzer,
            ICollector,
            IReporter,
        )

        # All should be non-None
        assert all([
            TrainingAdvisor,
            AdvisorConfig,
            TrainingMetrics,
            EpochMetrics,
            Issue,
            IssueType,
            IssueSeverity,
            Recommendation,
            RecommendationType,
            IAnalyzer,
            ICollector,
            IReporter,
        ])
