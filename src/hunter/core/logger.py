"""
Structured logging setup using structlog.

Follows SRP: Only responsible for logging concerns.
Follows DIP: Components depend on logger interface, not implementation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog

from .config import LoggingConfig


def setup_logger(config: LoggingConfig) -> structlog.BoundLogger:
    """
    Initialize and configure structured logger.

    Args:
        config: Logging configuration

    Returns:
        Configured structlog BoundLogger
    """
    # Set up standard logging level
    log_level = getattr(logging, config.level, logging.INFO)

    # Configure processors based on output format
    processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if configured
    if config.log_file:
        _setup_file_handler(config.log_file, log_level)

    return structlog.get_logger()


def _setup_file_handler(log_file: Path, level: int) -> None:
    """
    Set up file handler for logging.

    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Ensure parent directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add to root logger
    logging.getLogger().addHandler(handler)


class PipelineLogger:
    """
    Domain-specific logging helper for pipeline operations.

    Provides structured logging methods for common pipeline events.
    Follows Interface Segregation Principle (ISP):
    components only use the logging methods they need.
    """

    def __init__(self, logger: structlog.BoundLogger, config: LoggingConfig):
        self._log = logger
        self._config = config

    def frame_processed(
        self,
        frame_id: int,
        timings: dict[str, float],
        track_count: int,
        detection_count: int,
    ) -> None:
        """
        Log frame processing completion.

        Args:
            frame_id: Frame identifier
            timings: Stage timing dictionary
            track_count: Number of active tracks
            detection_count: Number of detections
        """
        if self._config.log_stage_timings:
            self._log.info(
                "frame_processed",
                frame_id=frame_id,
                timings=timings,
                tracks=track_count,
                detections=detection_count,
            )

    def track_state_change(
        self,
        track_id: int,
        old_state: str,
        new_state: str,
        reason: str,
    ) -> None:
        """
        Log track state transition.

        Args:
            track_id: Track identifier
            old_state: Previous state name
            new_state: New state name
            reason: Reason for transition
        """
        if self._config.log_track_changes:
            self._log.info(
                "track_state_change",
                track_id=track_id,
                from_state=old_state,
                to_state=new_state,
                reason=reason,
            )

    def track_created(self, track_id: int, bbox: tuple, confidence: float) -> None:
        """
        Log new track creation.

        Args:
            track_id: Track identifier
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Initial confidence
        """
        if self._config.log_track_changes:
            self._log.info(
                "track_created",
                track_id=track_id,
                bbox=list(bbox),
                confidence=round(confidence, 3),
            )

    def track_dropped(self, track_id: int, reason: str) -> None:
        """
        Log track termination.

        Args:
            track_id: Track identifier
            reason: Reason for dropping
        """
        if self._config.log_track_changes:
            self._log.info(
                "track_dropped",
                track_id=track_id,
                reason=reason,
            )

    def frame_dropped(self, frame_id: int, reason: str) -> None:
        """
        Log frame drop.

        Args:
            frame_id: Frame identifier
            reason: Reason for dropping
        """
        self._log.warning(
            "frame_dropped",
            frame_id=frame_id,
            reason=reason,
        )

    def model_loaded(self, model_name: str, model_hash: str, load_time_ms: float) -> None:
        """
        Log model loading.

        Args:
            model_name: Model name
            model_hash: Model checksum
            load_time_ms: Load time in milliseconds
        """
        self._log.info(
            "model_loaded",
            model_name=model_name,
            model_hash=model_hash[:16] + "...",
            load_time_ms=round(load_time_ms, 2),
        )

    def pipeline_started(self, config_summary: dict) -> None:
        """
        Log pipeline start.

        Args:
            config_summary: Summary of configuration
        """
        self._log.info(
            "pipeline_started",
            config=config_summary,
        )

    def pipeline_stopped(self, total_frames: int, runtime_seconds: float) -> None:
        """
        Log pipeline stop.

        Args:
            total_frames: Total frames processed
            runtime_seconds: Total runtime
        """
        self._log.info(
            "pipeline_stopped",
            total_frames=total_frames,
            runtime_seconds=round(runtime_seconds, 2),
            avg_fps=round(total_frames / runtime_seconds, 2) if runtime_seconds > 0 else 0,
        )

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error.

        Args:
            message: Error message
            **kwargs: Additional context
        """
        self._log.error(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning.

        Args:
            message: Warning message
            **kwargs: Additional context
        """
        self._log.warning(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info.

        Args:
            message: Info message
            **kwargs: Additional context
        """
        self._log.info(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug.

        Args:
            message: Debug message
            **kwargs: Additional context
        """
        self._log.debug(message, **kwargs)


def get_null_logger() -> "PipelineLogger":
    """
    Get a no-op logger for testing.

    Returns:
        PipelineLogger that does nothing
    """
    config = LoggingConfig(
        log_stage_timings=False,
        log_track_changes=False,
    )
    return PipelineLogger(structlog.get_logger(), config)
