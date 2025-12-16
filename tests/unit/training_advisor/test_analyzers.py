"""
Unit tests for training_advisor analyzers module.

Tests for rule-based analyzers.
"""

import pytest
from typing import List

from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics
from hunter.training_advisor.domain.issues import Issue, IssueType, IssueSeverity


# ============================================
# BaseAnalyzer Tests
# ============================================


class TestBaseAnalyzer:
    """Tests for BaseAnalyzer abstract class."""

    def test_class_exists(self):
        """BaseAnalyzer class exists."""
        from hunter.training_advisor.analyzers.base import BaseAnalyzer
        assert BaseAnalyzer is not None

    def test_implements_protocol(self):
        """BaseAnalyzer implements IAnalyzer protocol."""
        from hunter.training_advisor.analyzers.base import BaseAnalyzer
        from hunter.training_advisor.interfaces import IAnalyzer

        class TestAnalyzer(BaseAnalyzer):
            @property
            def name(self) -> str:
                return "test"

            def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
                return []

        analyzer = TestAnalyzer()
        assert isinstance(analyzer, IAnalyzer)


# ============================================
# OverfittingAnalyzer Tests
# ============================================


class TestOverfittingAnalyzer:
    """Tests for OverfittingAnalyzer."""

    def test_class_exists(self):
        """OverfittingAnalyzer class exists."""
        from hunter.training_advisor.analyzers.overfitting import OverfittingAnalyzer
        assert OverfittingAnalyzer is not None

    def test_name(self):
        """OverfittingAnalyzer has correct name."""
        from hunter.training_advisor.analyzers.overfitting import OverfittingAnalyzer

        analyzer = OverfittingAnalyzer()
        assert analyzer.name == "overfitting"

    def test_detects_overfitting(self):
        """OverfittingAnalyzer detects overfitting pattern."""
        from hunter.training_advisor.analyzers.overfitting import OverfittingAnalyzer

        # Create metrics showing overfitting (train_loss decreasing, val_loss increasing)
        epochs = [
            EpochMetrics(epoch=0, train_loss=1.0, val_loss=1.0),
            EpochMetrics(epoch=1, train_loss=0.8, val_loss=1.0),
            EpochMetrics(epoch=2, train_loss=0.6, val_loss=1.1),
            EpochMetrics(epoch=3, train_loss=0.4, val_loss=1.2),
            EpochMetrics(epoch=4, train_loss=0.3, val_loss=1.3),
            EpochMetrics(epoch=5, train_loss=0.2, val_loss=1.4),
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = OverfittingAnalyzer(gap_threshold=0.3, min_epochs=3)
        issues = analyzer.analyze(metrics)

        assert len(issues) > 0
        assert any(i.issue_type == IssueType.OVERFITTING for i in issues)

    def test_no_overfitting_when_converging(self):
        """OverfittingAnalyzer doesn't flag converging training."""
        from hunter.training_advisor.analyzers.overfitting import OverfittingAnalyzer

        # Create metrics showing normal convergence
        epochs = [
            EpochMetrics(epoch=0, train_loss=1.0, val_loss=1.1),
            EpochMetrics(epoch=1, train_loss=0.8, val_loss=0.9),
            EpochMetrics(epoch=2, train_loss=0.6, val_loss=0.7),
            EpochMetrics(epoch=3, train_loss=0.5, val_loss=0.6),
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = OverfittingAnalyzer()
        issues = analyzer.analyze(metrics)

        assert not any(i.issue_type == IssueType.OVERFITTING for i in issues)


# ============================================
# LearningRateAnalyzer Tests
# ============================================


class TestLearningRateAnalyzer:
    """Tests for LearningRateAnalyzer."""

    def test_class_exists(self):
        """LearningRateAnalyzer class exists."""
        from hunter.training_advisor.analyzers.learning_rate import LearningRateAnalyzer
        assert LearningRateAnalyzer is not None

    def test_name(self):
        """LearningRateAnalyzer has correct name."""
        from hunter.training_advisor.analyzers.learning_rate import LearningRateAnalyzer

        analyzer = LearningRateAnalyzer()
        assert analyzer.name == "learning_rate"

    def test_detects_lr_too_high(self):
        """LearningRateAnalyzer detects diverging loss."""
        from hunter.training_advisor.analyzers.learning_rate import LearningRateAnalyzer

        # Create metrics showing diverging loss
        epochs = [
            EpochMetrics(epoch=0, train_loss=1.0, val_loss=1.0, learning_rate=0.1),
            EpochMetrics(epoch=1, train_loss=1.5, val_loss=1.6, learning_rate=0.1),
            EpochMetrics(epoch=2, train_loss=2.5, val_loss=2.8, learning_rate=0.1),
            EpochMetrics(epoch=3, train_loss=5.0, val_loss=5.5, learning_rate=0.1),
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = LearningRateAnalyzer()
        issues = analyzer.analyze(metrics)

        assert len(issues) > 0
        assert any(i.issue_type in (IssueType.LR_TOO_HIGH, IssueType.DIVERGENCE) for i in issues)

    def test_detects_lr_too_low(self):
        """LearningRateAnalyzer detects slow learning."""
        from hunter.training_advisor.analyzers.learning_rate import LearningRateAnalyzer

        # Create metrics showing very slow learning
        epochs = [
            EpochMetrics(epoch=i, train_loss=1.0 - i * 0.001, val_loss=1.0 - i * 0.001, learning_rate=0.0001)
            for i in range(20)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = LearningRateAnalyzer(min_epochs_for_slow=10, slow_improvement_threshold=0.05)
        issues = analyzer.analyze(metrics)

        # Should detect slow progress
        assert any(i.issue_type == IssueType.LR_TOO_LOW for i in issues)


# ============================================
# ConvergenceAnalyzer Tests
# ============================================


class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer."""

    def test_class_exists(self):
        """ConvergenceAnalyzer class exists."""
        from hunter.training_advisor.analyzers.convergence import ConvergenceAnalyzer
        assert ConvergenceAnalyzer is not None

    def test_name(self):
        """ConvergenceAnalyzer has correct name."""
        from hunter.training_advisor.analyzers.convergence import ConvergenceAnalyzer

        analyzer = ConvergenceAnalyzer()
        assert analyzer.name == "convergence"

    def test_detects_plateau(self):
        """ConvergenceAnalyzer detects plateau."""
        from hunter.training_advisor.analyzers.convergence import ConvergenceAnalyzer

        # Create metrics showing plateau
        epochs = [
            EpochMetrics(epoch=i, train_loss=0.5, val_loss=0.6)
            for i in range(10)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = ConvergenceAnalyzer(plateau_epochs=5, min_improvement=0.001)
        issues = analyzer.analyze(metrics)

        assert any(i.issue_type == IssueType.PLATEAU for i in issues)


# ============================================
# DetectionAnalyzer Tests
# ============================================


class TestDetectionAnalyzer:
    """Tests for DetectionAnalyzer (YOLO-specific)."""

    def test_class_exists(self):
        """DetectionAnalyzer class exists."""
        from hunter.training_advisor.analyzers.detection import DetectionAnalyzer
        assert DetectionAnalyzer is not None

    def test_name(self):
        """DetectionAnalyzer has correct name."""
        from hunter.training_advisor.analyzers.detection import DetectionAnalyzer

        analyzer = DetectionAnalyzer()
        assert analyzer.name == "detection"

    def test_detects_low_map(self):
        """DetectionAnalyzer detects low mAP."""
        from hunter.training_advisor.analyzers.detection import DetectionAnalyzer

        epochs = [
            EpochMetrics(epoch=i, train_loss=0.5, val_loss=0.6, map50=0.2, map50_95=0.1)
            for i in range(10)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = DetectionAnalyzer(low_map_threshold=0.3)
        issues = analyzer.analyze(metrics)

        assert any(i.issue_type == IssueType.LOW_MAP for i in issues)

    def test_detects_precision_recall_imbalance(self):
        """DetectionAnalyzer detects precision-recall imbalance."""
        from hunter.training_advisor.analyzers.detection import DetectionAnalyzer

        epochs = [
            EpochMetrics(
                epoch=i, train_loss=0.5, val_loss=0.6,
                map50=0.5, precision=0.9, recall=0.3
            )
            for i in range(10)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = DetectionAnalyzer(imbalance_threshold=0.3)
        issues = analyzer.analyze(metrics)

        assert any(i.issue_type == IssueType.PRECISION_RECALL_IMBALANCE for i in issues)

    def test_no_issues_without_detection_metrics(self):
        """DetectionAnalyzer skips metrics without detection data."""
        from hunter.training_advisor.analyzers.detection import DetectionAnalyzer

        epochs = [
            EpochMetrics(epoch=i, train_loss=0.5, val_loss=0.6)  # No mAP
            for i in range(5)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = DetectionAnalyzer()
        issues = analyzer.analyze(metrics)

        # Should not have detection-specific issues
        assert not any(i.issue_type in (IssueType.LOW_MAP, IssueType.PRECISION_RECALL_IMBALANCE) for i in issues)


# ============================================
# SiameseAnalyzer Tests
# ============================================


class TestSiameseAnalyzer:
    """Tests for SiameseAnalyzer (embedding-specific)."""

    def test_class_exists(self):
        """SiameseAnalyzer class exists."""
        from hunter.training_advisor.analyzers.siamese import SiameseAnalyzer
        assert SiameseAnalyzer is not None

    def test_name(self):
        """SiameseAnalyzer has correct name."""
        from hunter.training_advisor.analyzers.siamese import SiameseAnalyzer

        analyzer = SiameseAnalyzer()
        assert analyzer.name == "siamese"

    def test_detects_triplet_loss_stuck(self):
        """SiameseAnalyzer detects stuck triplet loss."""
        from hunter.training_advisor.analyzers.siamese import SiameseAnalyzer

        epochs = [
            EpochMetrics(epoch=i, train_loss=0.5, val_loss=0.6, triplet_loss=0.5)
            for i in range(15)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        analyzer = SiameseAnalyzer(stuck_epochs=10, stuck_threshold=0.01)
        issues = analyzer.analyze(metrics)

        assert any(i.issue_type == IssueType.TRIPLET_LOSS_STUCK for i in issues)


# ============================================
# Analyzer Factory Tests
# ============================================


class TestAnalyzerFactory:
    """Tests for analyzer factory function."""

    def test_factory_exists(self):
        """get_analyzer factory function exists."""
        from hunter.training_advisor.analyzers import get_analyzer
        assert callable(get_analyzer)

    def test_get_all_analyzers(self):
        """get_all_analyzers returns all analyzers."""
        from hunter.training_advisor.analyzers import get_all_analyzers

        analyzers = get_all_analyzers()
        assert len(analyzers) >= 4  # At least overfitting, lr, convergence, detection

    def test_get_overfitting_analyzer(self):
        """Factory returns overfitting analyzer."""
        from hunter.training_advisor.analyzers import get_analyzer
        from hunter.training_advisor.analyzers.overfitting import OverfittingAnalyzer

        analyzer = get_analyzer("overfitting")
        assert isinstance(analyzer, OverfittingAnalyzer)

    def test_get_learning_rate_analyzer(self):
        """Factory returns learning rate analyzer."""
        from hunter.training_advisor.analyzers import get_analyzer
        from hunter.training_advisor.analyzers.learning_rate import LearningRateAnalyzer

        analyzer = get_analyzer("learning_rate")
        assert isinstance(analyzer, LearningRateAnalyzer)


# ============================================
# Analyzer Exports Tests
# ============================================


class TestAnalyzerExports:
    """Tests for analyzers module exports."""

    def test_exports_base_analyzer(self):
        """BaseAnalyzer is exported."""
        from hunter.training_advisor.analyzers import BaseAnalyzer
        assert BaseAnalyzer is not None

    def test_exports_overfitting_analyzer(self):
        """OverfittingAnalyzer is exported."""
        from hunter.training_advisor.analyzers import OverfittingAnalyzer
        assert OverfittingAnalyzer is not None

    def test_exports_learning_rate_analyzer(self):
        """LearningRateAnalyzer is exported."""
        from hunter.training_advisor.analyzers import LearningRateAnalyzer
        assert LearningRateAnalyzer is not None

    def test_exports_convergence_analyzer(self):
        """ConvergenceAnalyzer is exported."""
        from hunter.training_advisor.analyzers import ConvergenceAnalyzer
        assert ConvergenceAnalyzer is not None

    def test_exports_detection_analyzer(self):
        """DetectionAnalyzer is exported."""
        from hunter.training_advisor.analyzers import DetectionAnalyzer
        assert DetectionAnalyzer is not None

    def test_exports_siamese_analyzer(self):
        """SiameseAnalyzer is exported."""
        from hunter.training_advisor.analyzers import SiameseAnalyzer
        assert SiameseAnalyzer is not None
