"""
Unit tests for training_advisor domain module.

Tests for metrics, issues, and recommendations data classes.
"""

import pytest
from typing import List, Optional, Dict, Any
from enum import Enum


# ============================================
# TrainingMetrics Tests
# ============================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics data class."""

    def test_creation(self):
        """TrainingMetrics can be created."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        metrics = TrainingMetrics(epochs=[])
        assert metrics is not None
        assert metrics.epochs == []

    def test_with_epochs(self):
        """TrainingMetrics with epoch data."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        epoch = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
        )
        metrics = TrainingMetrics(epochs=[epoch])

        assert len(metrics.epochs) == 1
        assert metrics.epochs[0].epoch == 1

    def test_latest_epoch(self):
        """TrainingMetrics returns latest epoch."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        epochs = [
            EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6),
            EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.5),
            EpochMetrics(epoch=3, train_loss=0.3, val_loss=0.45),
        ]
        metrics = TrainingMetrics(epochs=epochs)

        latest = metrics.latest_epoch
        assert latest is not None
        assert latest.epoch == 3

    def test_best_epoch(self):
        """TrainingMetrics finds best epoch by val_loss."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        epochs = [
            EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6),
            EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.45),  # Best
            EpochMetrics(epoch=3, train_loss=0.3, val_loss=0.5),
        ]
        metrics = TrainingMetrics(epochs=epochs)

        best = metrics.best_epoch
        assert best is not None
        assert best.epoch == 2

    def test_empty_metrics(self):
        """TrainingMetrics handles empty epochs."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics

        metrics = TrainingMetrics(epochs=[])

        assert metrics.latest_epoch is None
        assert metrics.best_epoch is None

    def test_has_detection_metrics(self):
        """TrainingMetrics detects detection metrics presence."""
        from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics

        # Without detection metrics
        epoch_no_det = EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6)
        metrics_no_det = TrainingMetrics(epochs=[epoch_no_det])
        assert not metrics_no_det.has_detection_metrics

        # With detection metrics
        epoch_with_det = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            map50=0.7,
            map50_95=0.5,
        )
        metrics_with_det = TrainingMetrics(epochs=[epoch_with_det])
        assert metrics_with_det.has_detection_metrics


class TestEpochMetrics:
    """Tests for EpochMetrics data class."""

    def test_creation_minimal(self):
        """EpochMetrics with minimal fields."""
        from hunter.training_advisor.domain.metrics import EpochMetrics

        epoch = EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
        )
        assert epoch.epoch == 1
        assert epoch.train_loss == 0.5
        assert epoch.val_loss == 0.6

    def test_creation_full(self):
        """EpochMetrics with all fields."""
        from hunter.training_advisor.domain.metrics import EpochMetrics

        epoch = EpochMetrics(
            epoch=10,
            train_loss=0.1,
            val_loss=0.15,
            learning_rate=0.001,
            map50=0.85,
            map50_95=0.65,
            precision=0.9,
            recall=0.88,
            box_loss=0.02,
            cls_loss=0.03,
            dfl_loss=0.01,
            embedding_loss=0.05,
            triplet_loss=0.04,
        )

        assert epoch.learning_rate == 0.001
        assert epoch.map50 == 0.85
        assert epoch.precision == 0.9

    def test_loss_gap(self):
        """EpochMetrics calculates loss gap."""
        from hunter.training_advisor.domain.metrics import EpochMetrics

        epoch = EpochMetrics(epoch=1, train_loss=0.3, val_loss=0.5)

        assert epoch.loss_gap == pytest.approx(0.2)

    def test_is_improving(self):
        """EpochMetrics detects improvement."""
        from hunter.training_advisor.domain.metrics import EpochMetrics

        prev = EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6)
        curr = EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.55)

        assert curr.is_improving(prev)

        # Not improving
        worse = EpochMetrics(epoch=3, train_loss=0.35, val_loss=0.65)
        assert not worse.is_improving(curr)


# ============================================
# Issue Tests
# ============================================


class TestIssueType:
    """Tests for IssueType enum."""

    def test_enum_exists(self):
        """IssueType enum is importable."""
        from hunter.training_advisor.domain.issues import IssueType

        assert IssueType is not None

    def test_contains_expected_types(self):
        """IssueType contains all expected issue types."""
        from hunter.training_advisor.domain.issues import IssueType

        expected_types = [
            'OVERFITTING',
            'UNDERFITTING',
            'LR_TOO_HIGH',
            'LR_TOO_LOW',
            'PLATEAU',
            'DIVERGENCE',
            'GRADIENT_EXPLOSION',
            'LOW_MAP',
            'PRECISION_RECALL_IMBALANCE',
            'CLASS_IMBALANCE',
            'EMBEDDING_NOT_CONVERGING',
            'TRIPLET_LOSS_STUCK',
        ]

        for type_name in expected_types:
            assert hasattr(IssueType, type_name), f"Missing {type_name}"


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_enum_exists(self):
        """IssueSeverity enum is importable."""
        from hunter.training_advisor.domain.issues import IssueSeverity

        assert IssueSeverity is not None

    def test_contains_expected_levels(self):
        """IssueSeverity contains expected levels."""
        from hunter.training_advisor.domain.issues import IssueSeverity

        assert hasattr(IssueSeverity, 'LOW')
        assert hasattr(IssueSeverity, 'MEDIUM')
        assert hasattr(IssueSeverity, 'HIGH')
        assert hasattr(IssueSeverity, 'CRITICAL')


class TestIssue:
    """Tests for Issue data class."""

    def test_creation(self):
        """Issue can be created."""
        from hunter.training_advisor.domain.issues import Issue, IssueType, IssueSeverity

        issue = Issue(
            issue_type=IssueType.OVERFITTING,
            severity=IssueSeverity.HIGH,
            message="Overfitting detected",
            details={"val_loss_increase": 0.1},
            epoch_detected=10,
        )

        assert issue.issue_type == IssueType.OVERFITTING
        assert issue.severity == IssueSeverity.HIGH
        assert issue.message == "Overfitting detected"

    def test_to_dict(self):
        """Issue serializes to dict."""
        from hunter.training_advisor.domain.issues import Issue, IssueType, IssueSeverity

        issue = Issue(
            issue_type=IssueType.LR_TOO_HIGH,
            severity=IssueSeverity.MEDIUM,
            message="Learning rate too high",
            details={},
            epoch_detected=5,
        )

        d = issue.to_dict()
        assert isinstance(d, dict)
        assert 'issue_type' in d
        assert 'severity' in d
        assert 'message' in d

    def test_str_representation(self):
        """Issue has string representation."""
        from hunter.training_advisor.domain.issues import Issue, IssueType, IssueSeverity

        issue = Issue(
            issue_type=IssueType.PLATEAU,
            severity=IssueSeverity.LOW,
            message="Training plateaued",
            details={},
            epoch_detected=20,
        )

        s = str(issue)
        assert 'PLATEAU' in s or 'plateaued' in s.lower()


# ============================================
# Recommendation Tests
# ============================================


class TestRecommendationType:
    """Tests for RecommendationType enum."""

    def test_enum_exists(self):
        """RecommendationType enum is importable."""
        from hunter.training_advisor.domain.recommendations import RecommendationType

        assert RecommendationType is not None

    def test_contains_expected_types(self):
        """RecommendationType contains expected types."""
        from hunter.training_advisor.domain.recommendations import RecommendationType

        expected_types = [
            'REDUCE_LR',
            'INCREASE_LR',
            'ADD_REGULARIZATION',
            'INCREASE_DROPOUT',
            'EARLY_STOPPING',
            'INCREASE_DATA_AUGMENTATION',
            'REDUCE_MODEL_COMPLEXITY',
            'INCREASE_MODEL_CAPACITY',
            'ADJUST_BATCH_SIZE',
            'CHECK_DATA_QUALITY',
            'BALANCE_CLASSES',
            'ADJUST_ANCHOR_BOXES',
            'MODIFY_LOSS_WEIGHTS',
            'INCREASE_EPOCHS',
            'ADJUST_TRIPLET_MARGIN',
        ]

        for type_name in expected_types:
            assert hasattr(RecommendationType, type_name), f"Missing {type_name}"


class TestRecommendation:
    """Tests for Recommendation data class."""

    def test_creation(self):
        """Recommendation can be created."""
        from hunter.training_advisor.domain.recommendations import (
            Recommendation,
            RecommendationType,
        )
        from hunter.training_advisor.domain.issues import IssueType

        rec = Recommendation(
            rec_type=RecommendationType.REDUCE_LR,
            source_issue=IssueType.LR_TOO_HIGH,
            message="Reduce learning rate by 50%",
            suggested_value=0.0005,
            config_key="lr0",
            auto_applicable=True,
        )

        assert rec.rec_type == RecommendationType.REDUCE_LR
        assert rec.suggested_value == 0.0005
        assert rec.auto_applicable is True

    def test_to_dict(self):
        """Recommendation serializes to dict."""
        from hunter.training_advisor.domain.recommendations import (
            Recommendation,
            RecommendationType,
        )
        from hunter.training_advisor.domain.issues import IssueType

        rec = Recommendation(
            rec_type=RecommendationType.EARLY_STOPPING,
            source_issue=IssueType.OVERFITTING,
            message="Enable early stopping",
            suggested_value=None,
            config_key=None,
            auto_applicable=False,
        )

        d = rec.to_dict()
        assert isinstance(d, dict)
        assert 'rec_type' in d
        assert 'message' in d

    def test_priority(self):
        """Recommendation has priority property."""
        from hunter.training_advisor.domain.recommendations import (
            Recommendation,
            RecommendationType,
        )
        from hunter.training_advisor.domain.issues import IssueType

        rec = Recommendation(
            rec_type=RecommendationType.REDUCE_LR,
            source_issue=IssueType.DIVERGENCE,
            message="Reduce learning rate immediately",
            suggested_value=0.0001,
            config_key="lr0",
            auto_applicable=True,
            priority=1,
        )

        assert rec.priority == 1


# ============================================
# Domain Module Exports Tests
# ============================================


class TestDomainExports:
    """Tests for domain module exports."""

    def test_metrics_module_exports(self):
        """metrics module exports classes."""
        from hunter.training_advisor.domain import metrics

        assert hasattr(metrics, 'TrainingMetrics')
        assert hasattr(metrics, 'EpochMetrics')

    def test_issues_module_exports(self):
        """issues module exports classes."""
        from hunter.training_advisor.domain import issues

        assert hasattr(issues, 'Issue')
        assert hasattr(issues, 'IssueType')
        assert hasattr(issues, 'IssueSeverity')

    def test_recommendations_module_exports(self):
        """recommendations module exports classes."""
        from hunter.training_advisor.domain import recommendations

        assert hasattr(recommendations, 'Recommendation')
        assert hasattr(recommendations, 'RecommendationType')

    def test_domain_package_exports(self):
        """domain package exports all types."""
        from hunter.training_advisor.domain import (
            TrainingMetrics,
            EpochMetrics,
            Issue,
            IssueType,
            IssueSeverity,
            Recommendation,
            RecommendationType,
        )

        assert TrainingMetrics is not None
        assert EpochMetrics is not None
        assert Issue is not None
        assert IssueType is not None
        assert IssueSeverity is not None
        assert Recommendation is not None
        assert RecommendationType is not None
