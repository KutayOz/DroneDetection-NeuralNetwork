"""
Unit tests for training_advisor interfaces module.

Tests Protocol classes for analyzers, collectors, and reporters.
"""

import pytest
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


# ============================================
# IAnalyzer Protocol Tests
# ============================================


class TestIAnalyzerProtocol:
    """Tests for IAnalyzer Protocol interface."""

    def test_protocol_exists(self):
        """IAnalyzer Protocol is importable."""
        from hunter.training_advisor.interfaces.analyzer import IAnalyzer
        assert IAnalyzer is not None

    def test_protocol_is_runtime_checkable(self):
        """IAnalyzer is runtime checkable."""
        from hunter.training_advisor.interfaces.analyzer import IAnalyzer
        from typing import runtime_checkable, Protocol

        # Should be a Protocol
        assert hasattr(IAnalyzer, '__protocol_attrs__') or issubclass(IAnalyzer, Protocol)

    def test_analyze_method_signature(self):
        """IAnalyzer defines analyze method."""
        from hunter.training_advisor.interfaces.analyzer import IAnalyzer

        # Check method exists in protocol
        assert hasattr(IAnalyzer, 'analyze')

    def test_name_property(self):
        """IAnalyzer defines name property."""
        from hunter.training_advisor.interfaces.analyzer import IAnalyzer

        assert hasattr(IAnalyzer, 'name')

    def test_concrete_implementation_satisfies_protocol(self):
        """Concrete class satisfying IAnalyzer is accepted."""
        from hunter.training_advisor.interfaces.analyzer import IAnalyzer
        from hunter.training_advisor.domain.metrics import TrainingMetrics
        from hunter.training_advisor.domain.issues import Issue

        class ConcreteAnalyzer:
            @property
            def name(self) -> str:
                return "TestAnalyzer"

            def analyze(self, metrics: TrainingMetrics) -> List[Issue]:
                return []

        analyzer = ConcreteAnalyzer()
        assert isinstance(analyzer, IAnalyzer)


# ============================================
# ICollector Protocol Tests
# ============================================


class TestICollectorProtocol:
    """Tests for ICollector Protocol interface."""

    def test_protocol_exists(self):
        """ICollector Protocol is importable."""
        from hunter.training_advisor.interfaces.collector import ICollector
        assert ICollector is not None

    def test_collect_method_signature(self):
        """ICollector defines collect method."""
        from hunter.training_advisor.interfaces.collector import ICollector

        assert hasattr(ICollector, 'collect')

    def test_source_type_property(self):
        """ICollector defines source_type property."""
        from hunter.training_advisor.interfaces.collector import ICollector

        assert hasattr(ICollector, 'source_type')

    def test_concrete_implementation_satisfies_protocol(self):
        """Concrete class satisfying ICollector is accepted."""
        from hunter.training_advisor.interfaces.collector import ICollector
        from hunter.training_advisor.domain.metrics import TrainingMetrics
        from pathlib import Path

        class ConcreteCollector:
            @property
            def source_type(self) -> str:
                return "test"

            def collect(self, source: Path) -> TrainingMetrics:
                return TrainingMetrics(epochs=[])

        collector = ConcreteCollector()
        assert isinstance(collector, ICollector)


# ============================================
# IReporter Protocol Tests
# ============================================


class TestIReporterProtocol:
    """Tests for IReporter Protocol interface."""

    def test_protocol_exists(self):
        """IReporter Protocol is importable."""
        from hunter.training_advisor.interfaces.reporter import IReporter
        assert IReporter is not None

    def test_report_method_signature(self):
        """IReporter defines report method."""
        from hunter.training_advisor.interfaces.reporter import IReporter

        assert hasattr(IReporter, 'report')

    def test_format_property(self):
        """IReporter defines format property."""
        from hunter.training_advisor.interfaces.reporter import IReporter

        assert hasattr(IReporter, 'format')

    def test_concrete_implementation_satisfies_protocol(self):
        """Concrete class satisfying IReporter is accepted."""
        from hunter.training_advisor.interfaces.reporter import IReporter
        from hunter.training_advisor.domain.issues import Issue
        from hunter.training_advisor.domain.recommendations import Recommendation

        class ConcreteReporter:
            @property
            def format(self) -> str:
                return "console"

            def report(
                self,
                issues: List[Issue],
                recommendations: List[Recommendation],
                output_path: Optional[str] = None,
            ) -> str:
                return "Report output"

        reporter = ConcreteReporter()
        assert isinstance(reporter, IReporter)


# ============================================
# IAction Protocol Tests
# ============================================


class TestIActionProtocol:
    """Tests for IAction Protocol interface."""

    def test_protocol_exists(self):
        """IAction Protocol is importable."""
        from hunter.training_advisor.interfaces.analyzer import IAction
        assert IAction is not None

    def test_execute_method_signature(self):
        """IAction defines execute method."""
        from hunter.training_advisor.interfaces.analyzer import IAction

        assert hasattr(IAction, 'execute')

    def test_can_apply_method_signature(self):
        """IAction defines can_apply method."""
        from hunter.training_advisor.interfaces.analyzer import IAction

        assert hasattr(IAction, 'can_apply')

    def test_action_name_property(self):
        """IAction defines action_name property."""
        from hunter.training_advisor.interfaces.analyzer import IAction

        assert hasattr(IAction, 'action_name')

    def test_is_safe_property(self):
        """IAction defines is_safe property."""
        from hunter.training_advisor.interfaces.analyzer import IAction

        assert hasattr(IAction, 'is_safe')

    def test_concrete_implementation_satisfies_protocol(self):
        """Concrete class satisfying IAction is accepted."""
        from hunter.training_advisor.interfaces.analyzer import IAction
        from hunter.training_advisor.domain.recommendations import Recommendation

        class ConcreteAction:
            @property
            def action_name(self) -> str:
                return "test_action"

            @property
            def is_safe(self) -> bool:
                return True

            def can_apply(self, recommendation: Recommendation) -> bool:
                return True

            def execute(self, recommendation: Recommendation, config_path: str) -> bool:
                return True

        action = ConcreteAction()
        assert isinstance(action, IAction)


# ============================================
# ILLMProvider Protocol Tests
# ============================================


class TestILLMProviderProtocol:
    """Tests for ILLMProvider Protocol interface."""

    def test_protocol_exists(self):
        """ILLMProvider Protocol is importable."""
        from hunter.training_advisor.interfaces.analyzer import ILLMProvider
        assert ILLMProvider is not None

    def test_analyze_method_signature(self):
        """ILLMProvider defines analyze method."""
        from hunter.training_advisor.interfaces.analyzer import ILLMProvider

        assert hasattr(ILLMProvider, 'analyze')

    def test_provider_name_property(self):
        """ILLMProvider defines provider_name property."""
        from hunter.training_advisor.interfaces.analyzer import ILLMProvider

        assert hasattr(ILLMProvider, 'provider_name')

    def test_concrete_implementation_satisfies_protocol(self):
        """Concrete class satisfying ILLMProvider is accepted."""
        from hunter.training_advisor.interfaces.analyzer import ILLMProvider
        from hunter.training_advisor.domain.metrics import TrainingMetrics
        from hunter.training_advisor.domain.issues import Issue

        class ConcreteLLMProvider:
            @property
            def provider_name(self) -> str:
                return "test_llm"

            def analyze(
                self,
                metrics: TrainingMetrics,
                issues: List[Issue],
                context: Optional[str] = None,
            ) -> str:
                return "LLM analysis"

        provider = ConcreteLLMProvider()
        assert isinstance(provider, ILLMProvider)


# ============================================
# Interface Module Exports Tests
# ============================================


class TestInterfaceExports:
    """Tests for interfaces module exports."""

    def test_analyzer_module_exports(self):
        """analyzer module exports all interfaces."""
        from hunter.training_advisor.interfaces import analyzer

        assert hasattr(analyzer, 'IAnalyzer')
        assert hasattr(analyzer, 'IAction')
        assert hasattr(analyzer, 'ILLMProvider')

    def test_collector_module_exports(self):
        """collector module exports ICollector."""
        from hunter.training_advisor.interfaces import collector

        assert hasattr(collector, 'ICollector')

    def test_reporter_module_exports(self):
        """reporter module exports IReporter."""
        from hunter.training_advisor.interfaces import reporter

        assert hasattr(reporter, 'IReporter')

    def test_interfaces_package_exports(self):
        """interfaces package exports all protocols."""
        from hunter.training_advisor.interfaces import (
            IAnalyzer,
            ICollector,
            IReporter,
            IAction,
            ILLMProvider,
        )

        assert IAnalyzer is not None
        assert ICollector is not None
        assert IReporter is not None
        assert IAction is not None
        assert ILLMProvider is not None
