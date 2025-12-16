"""
Core infrastructure modules.

Provides foundational components:
- Configuration management
- Exception hierarchy
- Logging infrastructure
- Timing and metrics
"""

from .config import (
    DetectorConfig,
    EmbedderConfig,
    HunterConfig,
    IngestConfig,
    LoggingConfig,
    OutputConfig,
    PreprocessConfig,
    TrackingConfig,
)
from .exceptions import (
    ConfigError,
    HunterError,
    IngestError,
    ModelError,
    OutputError,
    PreprocessError,
    TrackingError,
)
from .logger import PipelineLogger, get_null_logger, setup_logger
from .metrics import LatencyMetrics, MetricsCollector, ThroughputMetrics, TrackingMetrics
from .timer import PipelineTimer, ScopedTimer, StageTimings

__all__ = [
    # Config
    "HunterConfig",
    "IngestConfig",
    "PreprocessConfig",
    "DetectorConfig",
    "EmbedderConfig",
    "TrackingConfig",
    "OutputConfig",
    "LoggingConfig",
    # Exceptions
    "HunterError",
    "ConfigError",
    "IngestError",
    "ModelError",
    "TrackingError",
    "PreprocessError",
    "OutputError",
    # Logger
    "setup_logger",
    "PipelineLogger",
    "get_null_logger",
    # Timer
    "PipelineTimer",
    "StageTimings",
    "ScopedTimer",
    # Metrics
    "LatencyMetrics",
    "ThroughputMetrics",
    "TrackingMetrics",
    "MetricsCollector",
]
