"""
Simple Dependency Injection container.

Provides centralized dependency management following DIP:
- All dependencies are registered here
- Components receive dependencies through constructor injection
- Easy testing through dependency overrides

Follows Dependency Inversion Principle (DIP):
- High-level modules depend on abstractions
- Low-level modules provide concrete implementations
- Container wires everything together
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from ..core.config import HunterConfig
from ..core.timer import PipelineTimer
from ..core.metrics import MetricsCollector

from ..detection.stub import StubDetector, StubEmbedder, StubVerifier

from ..pipeline.ingest import StubIngest
from ..pipeline.output import StubSink


T = TypeVar('T')


class Provider:
    """
    Simple provider class for lazy instantiation.

    Supports singleton and factory patterns.
    """

    def __init__(
        self,
        factory: Callable[..., T],
        singleton: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize provider.

        Args:
            factory: Factory function or class to instantiate
            singleton: If True, cache the instance
            **kwargs: Arguments to pass to factory
        """
        self._factory = factory
        self._singleton = singleton
        self._kwargs = kwargs
        self._instance: Optional[T] = None

    def __call__(self, **override_kwargs: Any) -> T:
        """
        Get or create instance.

        Args:
            **override_kwargs: Arguments to override defaults

        Returns:
            Instance of type T
        """
        if self._singleton and self._instance is not None:
            return self._instance

        # Merge kwargs with overrides
        kwargs = {**self._kwargs, **override_kwargs}

        # Resolve any Provider values in kwargs
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, Provider):
                resolved[key] = value()
            else:
                resolved[key] = value

        instance = self._factory(**resolved)

        if self._singleton:
            self._instance = instance

        return instance

    def override(self, factory: Callable[..., T]) -> None:
        """
        Override the factory for testing.

        Args:
            factory: New factory function
        """
        self._factory = factory
        self._instance = None

    def reset(self) -> None:
        """Reset singleton instance."""
        self._instance = None


class HunterContainer:
    """
    Main DI container for Hunter Drone system.

    Provides all dependencies through constructor injection.
    Supports overriding for testing.

    Example usage:
        container = HunterContainer(config)
        detector = container.detector()
        tracker = container.tracker()
    """

    def __init__(self, config: Optional[HunterConfig] = None) -> None:
        """
        Initialize container with configuration.

        Args:
            config: Hunter configuration (uses defaults if None)
        """
        self._config = config
        self._providers: Dict[str, Provider] = {}
        self._setup_providers()

    def _setup_providers(self) -> None:
        """Setup all dependency providers."""
        # Core services
        self._providers['timer'] = Provider(PipelineTimer)
        self._providers['metrics'] = Provider(MetricsCollector, singleton=True)

        # Detection components (lazy loaded when config available)
        # These will be setup when detector/embedder properties are accessed

    @property
    def config(self) -> Optional[HunterConfig]:
        """Get configuration."""
        return self._config

    def timer(self) -> PipelineTimer:
        """Get new timer instance."""
        return self._providers['timer']()

    def metrics(self) -> MetricsCollector:
        """Get metrics collector singleton."""
        return self._providers['metrics']()

    def detector(self):
        """
        Get detector instance.

        Requires config to be set with detector configuration.
        """
        if self._config is None:
            raise ValueError("Config not set - use create_container() or set config")

        from ..detection.yolo11_detector import YOLO11Detector
        return YOLO11Detector(
            model_path=self._config.detector.model_path,
            confidence_threshold=self._config.detector.confidence_threshold,
            nms_threshold=self._config.detector.nms_threshold,
            max_detections=self._config.detector.max_detections,
            device=self._config.detector.device,
        )

    def embedder(self):
        """
        Get embedder instance.

        Requires config to be set with embedder configuration.
        """
        if self._config is None:
            raise ValueError("Config not set")

        if not self._config.embedder.enabled or not self._config.embedder.model_path:
            return None

        from ..detection.siamese_embedder import SiameseEmbedder
        return SiameseEmbedder(
            model_path=self._config.embedder.model_path,
            embedding_dim=self._config.embedder.embedding_dim,
            input_size=self._config.embedder.input_size,
            device=self._config.embedder.device,
        )


class StubContainer:
    """
    Stub DI container with stub implementations.

    Used for unit and integration testing without real models.
    All components are stubs that don't require real model files.

    Also exported as TestContainer for backwards compatibility.
    """

    def __init__(self) -> None:
        """Initialize test container with stub providers."""
        self._detector_instance: Optional[StubDetector] = None
        self._embedder_instance: Optional[StubEmbedder] = None
        self._verifier_instance: Optional[StubVerifier] = None

    def detector(self) -> StubDetector:
        """Get stub detector (singleton)."""
        if self._detector_instance is None:
            self._detector_instance = StubDetector()
        return self._detector_instance

    def embedder(self) -> StubEmbedder:
        """Get stub embedder (singleton)."""
        if self._embedder_instance is None:
            self._embedder_instance = StubEmbedder(embedding_dim=128)
        return self._embedder_instance

    def verifier(self) -> StubVerifier:
        """Get stub verifier (singleton)."""
        if self._verifier_instance is None:
            self._verifier_instance = StubVerifier(always_verify=True, fixed_score=0.9)
        return self._verifier_instance

    def ingest(
        self,
        num_frames: int = 100,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> StubIngest:
        """
        Get stub ingest (factory - new instance each call).

        Args:
            num_frames: Number of frames to generate
            width: Frame width
            height: Frame height
            fps: Frame rate

        Returns:
            New StubIngest instance
        """
        return StubIngest(
            num_frames=num_frames,
            width=width,
            height=height,
            fps=fps,
        )

    def sink(self) -> StubSink:
        """Get stub sink (factory - new instance each call)."""
        return StubSink()


def create_container(config_path: Optional[Path] = None) -> HunterContainer:
    """
    Create and configure the DI container.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        Configured HunterContainer instance
    """
    config = None
    if config_path:
        config = HunterConfig.from_yaml(config_path)

    return HunterContainer(config)


def create_test_container() -> StubContainer:
    """
    Create a test container with all stubs.

    Returns:
        StubContainer with stub implementations
    """
    return StubContainer()


# Alias for backwards compatibility
TestContainer = StubContainer
