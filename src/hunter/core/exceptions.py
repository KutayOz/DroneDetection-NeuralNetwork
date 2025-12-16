"""
Custom exception classes for Hunter Drone system.

Each exception class follows Single Responsibility Principle (SRP):
- One exception type per error domain
- Clear hierarchy for error handling
"""

from typing import Optional


class HunterError(Exception):
    """
    Base exception for all Hunter system errors.

    All custom exceptions inherit from this class,
    allowing catch-all handling when needed.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigError(HunterError):
    """
    Configuration related errors.

    Raised when:
    - Config file not found
    - Invalid config values
    - Schema validation fails
    """
    pass


class IngestError(HunterError):
    """
    Video ingest errors.

    Raised when:
    - Video source not found
    - Failed to open stream
    - Decode errors
    - Timeout waiting for frames
    """
    pass


class ModelError(HunterError):
    """
    Model loading and inference errors.

    Raised when:
    - Model file not found
    - Invalid model format
    - Checksum mismatch
    - Inference failure
    """
    pass


class TrackingError(HunterError):
    """
    Tracking related errors.

    Raised when:
    - Invalid track state
    - Association failure
    - Kalman filter errors
    """
    pass


class PreprocessError(HunterError):
    """
    Preprocessing errors.

    Raised when:
    - Invalid image format
    - Resize failure
    - Color conversion error
    """
    pass


class OutputError(HunterError):
    """
    Output sink errors.

    Raised when:
    - Failed to write output
    - Invalid output format
    - Connection errors (for network sinks)
    """
    pass
