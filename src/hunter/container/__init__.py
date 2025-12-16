"""
Dependency Injection container using dependency-injector.

Provides centralized dependency management following DIP:
- All dependencies are registered here
- Components receive dependencies through constructor injection
- Easy testing through dependency overrides

Container Structure:
    HunterContainer
    ├── ConfigProvider
    ├── LoggerProvider
    ├── DetectorProvider
    │   ├── PrimaryDetectorProvider (YOLO11)
    │   └── SecondaryDetectorProvider (Siamese)
    ├── EmbedderProvider
    ├── TrackerProvider
    └── PipelineProvider

Example usage:
    from hunter.container import HunterContainer, create_container

    # Production usage
    container = create_container(Path("config.yaml"))
    detector = container.detector()

    # Testing usage
    from hunter.container import TestContainer, create_test_container
    container = create_test_container()
"""

from .container import (
    HunterContainer,
    TestContainer,
    create_container,
    create_test_container,
)

__all__ = [
    "HunterContainer",
    "TestContainer",
    "create_container",
    "create_test_container",
]
