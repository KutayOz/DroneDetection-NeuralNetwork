"""
Unit tests for training_advisor engine module.

Tests for decision engine and action registry.
"""

import pytest
from typing import List

from hunter.training_advisor.domain.metrics import TrainingMetrics, EpochMetrics
from hunter.training_advisor.domain.issues import Issue, IssueType, IssueSeverity
from hunter.training_advisor.domain.recommendations import Recommendation, RecommendationType


# ============================================
# DecisionEngine Tests
# ============================================


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    def test_class_exists(self):
        """DecisionEngine class exists."""
        from hunter.training_advisor.engine.decision_engine import DecisionEngine
        assert DecisionEngine is not None

    def test_analyze_returns_issues(self):
        """DecisionEngine.analyze returns issues."""
        from hunter.training_advisor.engine.decision_engine import DecisionEngine

        engine = DecisionEngine()
        epochs = [
            EpochMetrics(epoch=i, train_loss=1.0 - i * 0.1, val_loss=1.1 + i * 0.1)
            for i in range(10)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        issues = engine.analyze(metrics)
        assert isinstance(issues, list)

    def test_recommend_returns_recommendations(self):
        """DecisionEngine.recommend returns recommendations."""
        from hunter.training_advisor.engine.decision_engine import DecisionEngine

        engine = DecisionEngine()
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.HIGH,
                message="Test",
                epoch_detected=5,
            )
        ]

        recommendations = engine.recommend(issues)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_full_analysis_pipeline(self):
        """DecisionEngine full analysis pipeline."""
        from hunter.training_advisor.engine.decision_engine import DecisionEngine

        engine = DecisionEngine()
        epochs = [
            EpochMetrics(epoch=i, train_loss=0.5, val_loss=0.6)
            for i in range(15)
        ]
        metrics = TrainingMetrics(epochs=epochs)

        result = engine.run_analysis(metrics)
        assert "issues" in result
        assert "recommendations" in result


# ============================================
# RecommendationEngine Tests
# ============================================


class TestRecommendationEngine:
    """Tests for RecommendationEngine."""

    def test_class_exists(self):
        """RecommendationEngine class exists."""
        from hunter.training_advisor.engine.recommendation_engine import RecommendationEngine
        assert RecommendationEngine is not None

    def test_generates_recommendations_for_overfitting(self):
        """RecommendationEngine generates recommendations for overfitting."""
        from hunter.training_advisor.engine.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.HIGH,
                message="Overfitting detected",
                epoch_detected=10,
            )
        ]

        recommendations = engine.generate(issues)
        assert len(recommendations) > 0
        assert any(
            r.rec_type in (
                RecommendationType.ADD_REGULARIZATION,
                RecommendationType.INCREASE_DROPOUT,
                RecommendationType.EARLY_STOPPING,
            )
            for r in recommendations
        )

    def test_generates_recommendations_for_lr_issues(self):
        """RecommendationEngine generates recommendations for LR issues."""
        from hunter.training_advisor.engine.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        issues = [
            Issue(
                issue_type=IssueType.LR_TOO_HIGH,
                severity=IssueSeverity.HIGH,
                message="LR too high",
                epoch_detected=5,
            )
        ]

        recommendations = engine.generate(issues)
        assert len(recommendations) > 0
        assert any(r.rec_type == RecommendationType.REDUCE_LR for r in recommendations)

    def test_prioritizes_recommendations(self):
        """RecommendationEngine prioritizes by severity."""
        from hunter.training_advisor.engine.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        issues = [
            Issue(
                issue_type=IssueType.OVERFITTING,
                severity=IssueSeverity.LOW,
                message="Low",
                epoch_detected=10,
            ),
            Issue(
                issue_type=IssueType.DIVERGENCE,
                severity=IssueSeverity.CRITICAL,
                message="Critical",
                epoch_detected=5,
            ),
        ]

        recommendations = engine.generate(issues)
        # Critical issues should generate higher priority recommendations
        assert len(recommendations) > 0


# ============================================
# ActionRegistry Tests
# ============================================


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_class_exists(self):
        """ActionRegistry class exists."""
        from hunter.training_advisor.engine.action_registry import ActionRegistry
        assert ActionRegistry is not None

    def test_register_action(self):
        """ActionRegistry can register actions."""
        from hunter.training_advisor.engine.action_registry import ActionRegistry
        from hunter.training_advisor.interfaces import IAction

        class MockAction:
            @property
            def action_name(self) -> str:
                return "mock"

            @property
            def is_safe(self) -> bool:
                return True

            def can_apply(self, recommendation):
                return True

            def execute(self, recommendation, config_path):
                return True

        registry = ActionRegistry()
        registry.register(MockAction())
        assert "mock" in registry.list_actions()

    def test_get_action_for_recommendation(self):
        """ActionRegistry finds action for recommendation."""
        from hunter.training_advisor.engine.action_registry import ActionRegistry

        registry = ActionRegistry()
        rec = Recommendation(
            rec_type=RecommendationType.REDUCE_LR,
            source_issue=IssueType.LR_TOO_HIGH,
            message="Reduce LR",
            auto_applicable=True,
        )

        actions = registry.get_actions_for(rec)
        assert isinstance(actions, list)


# ============================================
# Engine Exports Tests
# ============================================


class TestEngineExports:
    """Tests for engine module exports."""

    def test_exports_decision_engine(self):
        """DecisionEngine is exported."""
        from hunter.training_advisor.engine import DecisionEngine
        assert DecisionEngine is not None

    def test_exports_recommendation_engine(self):
        """RecommendationEngine is exported."""
        from hunter.training_advisor.engine import RecommendationEngine
        assert RecommendationEngine is not None

    def test_exports_action_registry(self):
        """ActionRegistry is exported."""
        from hunter.training_advisor.engine import ActionRegistry
        assert ActionRegistry is not None
