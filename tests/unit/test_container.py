"""
Unit tests for DI container module.

Tests for HunterContainer and TestContainer.
"""

import pytest

from hunter.container.container import (
    HunterContainer,
    TestContainer,
    create_container,
    create_test_container,
    Provider,
)
from hunter.detection.stub import StubDetector, StubEmbedder, StubVerifier
from hunter.pipeline.ingest import StubIngest
from hunter.pipeline.output import StubSink
from hunter.core.timer import PipelineTimer
from hunter.core.metrics import MetricsCollector


# ============================================
# Provider Tests
# ============================================


class TestProvider:
    """Tests for Provider class."""

    def test_factory_creates_instance(self):
        """Provider creates instance from factory."""
        provider = Provider(PipelineTimer)
        instance = provider()
        assert isinstance(instance, PipelineTimer)

    def test_factory_creates_new_each_time(self):
        """Non-singleton provider creates new instance each call."""
        provider = Provider(PipelineTimer)
        instance1 = provider()
        instance2 = provider()
        assert instance1 is not instance2

    def test_singleton_returns_same_instance(self):
        """Singleton provider returns same instance."""
        provider = Provider(MetricsCollector, singleton=True)
        instance1 = provider()
        instance2 = provider()
        assert instance1 is instance2

    def test_kwargs_passed_to_factory(self):
        """Provider passes kwargs to factory."""
        provider = Provider(StubEmbedder, embedding_dim=256)
        instance = provider()
        assert instance.embedding_dim == 256

    def test_override_kwargs(self):
        """Override kwargs when calling provider."""
        provider = Provider(StubEmbedder, embedding_dim=128)
        instance = provider(embedding_dim=256)
        assert instance.embedding_dim == 256

    def test_reset_clears_singleton(self):
        """reset() clears singleton instance."""
        provider = Provider(MetricsCollector, singleton=True)
        instance1 = provider()
        provider.reset()
        instance2 = provider()
        assert instance1 is not instance2


# ============================================
# TestContainer Tests
# ============================================


class TestTestContainer:
    """Tests for TestContainer (stub container for testing)."""

    def test_creates_stub_detector(self):
        """TestContainer provides stub detector."""
        container = TestContainer()
        detector = container.detector()
        assert isinstance(detector, StubDetector)

    def test_detector_is_singleton(self):
        """Detector is singleton within container."""
        container = TestContainer()
        detector1 = container.detector()
        detector2 = container.detector()
        assert detector1 is detector2

    def test_creates_stub_embedder(self):
        """TestContainer provides stub embedder."""
        container = TestContainer()
        embedder = container.embedder()
        assert isinstance(embedder, StubEmbedder)
        assert embedder.embedding_dim == 128

    def test_creates_stub_verifier(self):
        """TestContainer provides stub verifier."""
        container = TestContainer()
        verifier = container.verifier()
        assert isinstance(verifier, StubVerifier)

    def test_creates_stub_ingest(self):
        """TestContainer provides stub ingest."""
        container = TestContainer()
        ingest = container.ingest()
        assert isinstance(ingest, StubIngest)

    def test_ingest_is_factory(self):
        """Ingest creates new instance each call."""
        container = TestContainer()
        ingest1 = container.ingest()
        ingest2 = container.ingest()
        assert ingest1 is not ingest2

    def test_ingest_custom_params(self):
        """Ingest accepts custom parameters."""
        container = TestContainer()
        ingest = container.ingest(num_frames=50, width=320, height=240, fps=15.0)
        assert ingest.frame_count == 50
        assert ingest.fps == 15.0

    def test_creates_stub_sink(self):
        """TestContainer provides stub sink."""
        container = TestContainer()
        sink = container.sink()
        assert isinstance(sink, StubSink)


# ============================================
# HunterContainer Tests
# ============================================


class TestHunterContainer:
    """Tests for HunterContainer."""

    def test_creates_timer(self):
        """HunterContainer creates timer."""
        container = HunterContainer()
        timer = container.timer()
        assert isinstance(timer, PipelineTimer)

    def test_creates_metrics(self):
        """HunterContainer creates metrics collector."""
        container = HunterContainer()
        metrics = container.metrics()
        assert isinstance(metrics, MetricsCollector)

    def test_metrics_is_singleton(self):
        """Metrics collector is singleton."""
        container = HunterContainer()
        metrics1 = container.metrics()
        metrics2 = container.metrics()
        assert metrics1 is metrics2

    def test_detector_requires_config(self):
        """detector() raises error without config."""
        container = HunterContainer()
        with pytest.raises(ValueError, match="Config not set"):
            container.detector()


# ============================================
# Factory Function Tests
# ============================================


class TestCreateTestContainer:
    """Tests for create_test_container function."""

    def test_creates_container(self):
        """create_test_container() returns TestContainer."""
        container = create_test_container()
        assert isinstance(container, TestContainer)

    def test_container_has_all_components(self):
        """Container has all required components."""
        container = create_test_container()

        # Check all methods exist
        assert hasattr(container, 'detector')
        assert hasattr(container, 'embedder')
        assert hasattr(container, 'verifier')
        assert hasattr(container, 'ingest')
        assert hasattr(container, 'sink')


class TestCreateContainer:
    """Tests for create_container function."""

    def test_creates_container(self):
        """create_container() returns HunterContainer."""
        container = create_container()
        assert isinstance(container, HunterContainer)

    def test_creates_without_config(self):
        """create_container() works without config path."""
        container = create_container(config_path=None)
        assert container.config is None


# ============================================
# Container Exports Tests
# ============================================


class TestContainerExports:
    """Tests for container module exports."""

    def test_exports_hunter_container(self):
        """HunterContainer is exported."""
        from hunter.container import HunterContainer
        assert HunterContainer is not None

    def test_exports_test_container(self):
        """TestContainer is exported."""
        from hunter.container import TestContainer
        assert TestContainer is not None

    def test_exports_create_container(self):
        """create_container is exported."""
        from hunter.container import create_container
        assert callable(create_container)

    def test_exports_create_test_container(self):
        """create_test_container is exported."""
        from hunter.container import create_test_container
        assert callable(create_test_container)
