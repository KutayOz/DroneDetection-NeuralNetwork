"""
Custom exception classes for Hunter Drone system.

Each exception class follows Single Responsibility Principle (SRP):
- One exception type per error domain
- Clear hierarchy for error handling
- Actionable error messages with suggestions
"""

from pathlib import Path
from typing import Optional, List


class HunterError(Exception):
    """
    Base exception for all Hunter system errors.

    All custom exceptions inherit from this class,
    allowing catch-all handling when needed.

    Features:
    - Stores error message and optional details dict
    - Generates suggestions for common issues
    - Formats nicely for console output
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"  Details: {details_str}")

        if self.suggestions:
            parts.append("")
            parts.append("  Suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"    {i}. {suggestion}")

        return "\n".join(parts)

    def add_suggestion(self, suggestion: str) -> "HunterError":
        """Add a suggestion to the error."""
        self.suggestions.append(suggestion)
        return self


class ConfigError(HunterError):
    """
    Configuration related errors.

    Raised when:
    - Config file not found
    - Invalid config values
    - Schema validation fails
    """

    @classmethod
    def file_not_found(cls, path: Path) -> "ConfigError":
        """Create error for missing config file."""
        return cls(
            f"Configuration file not found: {path}",
            details={"path": str(path)},
            suggestions=[
                f"Check that the file exists: ls {path}",
                "Copy the default config: cp configs/default.yaml configs/my_config.yaml",
                "Verify the path is correct (use absolute path if unsure)",
            ],
        )

    @classmethod
    def invalid_yaml(cls, path: Path, error: str) -> "ConfigError":
        """Create error for invalid YAML syntax."""
        return cls(
            f"Invalid YAML syntax in {path}",
            details={"path": str(path), "error": error},
            suggestions=[
                "Check for proper indentation (use spaces, not tabs)",
                "Ensure all colons have a space after them",
                "Verify quotes are properly closed",
                "Use a YAML validator: python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"",
            ],
        )

    @classmethod
    def validation_failed(cls, field: str, value: any, reason: str) -> "ConfigError":
        """Create error for field validation failure."""
        return cls(
            f"Invalid configuration value for '{field}': {reason}",
            details={"field": field, "value": value},
            suggestions=[
                f"Check the valid range for '{field}' in docs/configuration.md",
                "Use 'hunter validate config.yaml' to check your config",
            ],
        )


class IngestError(HunterError):
    """
    Video ingest errors.

    Raised when:
    - Video source not found
    - Failed to open stream
    - Decode errors
    - Timeout waiting for frames
    """

    @classmethod
    def source_not_found(cls, source: str) -> "IngestError":
        """Create error for missing video source."""
        suggestions = [
            f"Check that the file/path exists: ls {source}",
            "Verify the path is correct",
        ]

        # Add context-specific suggestions
        if source.startswith("rtsp://"):
            suggestions.extend([
                "Check that the RTSP server is running",
                "Verify network connectivity: ping <server-ip>",
                "Test the stream: ffplay " + source,
            ])
        elif source.isdigit():
            suggestions.extend([
                f"Check that camera {source} is connected",
                "List available cameras: ls /dev/video* (Linux)",
            ])
        else:
            suggestions.append("Supported formats: MP4, AVI, MKV, MOV")

        return cls(
            f"Video source not found or cannot be opened: {source}",
            details={"source": source},
            suggestions=suggestions,
        )

    @classmethod
    def stream_timeout(cls, source: str, timeout_ms: int) -> "IngestError":
        """Create error for stream timeout."""
        return cls(
            f"Timeout waiting for frames from: {source}",
            details={"source": source, "timeout_ms": timeout_ms},
            suggestions=[
                "Check network connectivity to the stream source",
                f"Increase timeout in config: ingest.timeout_ms: {timeout_ms * 2}",
                "Verify the stream is active: ffprobe " + source,
            ],
        )

    @classmethod
    def decode_error(cls, source: str, error: str) -> "IngestError":
        """Create error for video decode failure."""
        return cls(
            f"Failed to decode video from: {source}",
            details={"source": source, "error": error},
            suggestions=[
                "The video file may be corrupted",
                "Try re-encoding: ffmpeg -i input.mp4 -c:v libx264 output.mp4",
                "Check codec support: ffprobe " + source,
            ],
        )


class ModelError(HunterError):
    """
    Model loading and inference errors.

    Raised when:
    - Model file not found
    - Invalid model format
    - Checksum mismatch
    - Inference failure
    """

    @classmethod
    def model_not_found(cls, path: Path) -> "ModelError":
        """Create error for missing model file."""
        return cls(
            f"Model file not found: {path}",
            details={"path": str(path)},
            suggestions=[
                "Download a pre-trained model:",
                "  python -c \"from ultralytics import YOLO; YOLO('yolo11m.pt')\"",
                f"  mv yolo11m.pt {path.parent}/",
                "Or train your own model: hunter train --data datasets/drones",
            ],
        )

    @classmethod
    def invalid_format(cls, path: Path, expected: str) -> "ModelError":
        """Create error for invalid model format."""
        return cls(
            f"Invalid model format: {path.suffix}",
            details={"path": str(path), "expected": expected},
            suggestions=[
                f"Supported formats: .pt, .onnx, .engine",
                "Convert model format: yolo export model=model.pt format=onnx",
            ],
        )

    @classmethod
    def inference_failed(cls, model_name: str, error: str) -> "ModelError":
        """Create error for inference failure."""
        suggestions = [
            "Check GPU memory: nvidia-smi",
            "Try reducing batch size or image size",
            "Use CPU fallback: --device cpu",
        ]

        if "CUDA" in error or "cuda" in error:
            suggestions.insert(0, "CUDA error - check GPU drivers and CUDA installation")

        return cls(
            f"Model inference failed: {model_name}",
            details={"model": model_name, "error": error},
            suggestions=suggestions,
        )

    @classmethod
    def out_of_memory(cls, model_name: str, required_mb: int = 0) -> "ModelError":
        """Create error for GPU out of memory."""
        return cls(
            f"GPU out of memory while loading model: {model_name}",
            details={"model": model_name},
            suggestions=[
                "Use a smaller model: detector.model_path: models/yolo11n.pt",
                "Reduce input resolution: preprocess.input_size: [416, 416]",
                "Enable half precision: detector.half_precision: true",
                "Use CPU: detector.device: cpu",
                "Close other GPU applications",
            ],
        )


class TrackingError(HunterError):
    """
    Tracking related errors.

    Raised when:
    - Invalid track state
    - Association failure
    - Kalman filter errors
    """

    @classmethod
    def invalid_state_transition(
        cls, track_id: int, from_state: str, to_state: str
    ) -> "TrackingError":
        """Create error for invalid state transition."""
        return cls(
            f"Invalid state transition for track {track_id}: {from_state} -> {to_state}",
            details={"track_id": track_id, "from": from_state, "to": to_state},
            suggestions=[
                "This is likely a bug in the tracking logic",
                "Please report this issue with reproduction steps",
            ],
        )

    @classmethod
    def kalman_error(cls, error: str) -> "TrackingError":
        """Create error for Kalman filter issues."""
        return cls(
            f"Kalman filter error: {error}",
            suggestions=[
                "Try adjusting noise parameters:",
                "  tracking.process_noise: 1.0 (increase for more responsiveness)",
                "  tracking.measurement_noise: 1.0 (increase for smoother tracks)",
            ],
        )


class PreprocessError(HunterError):
    """
    Preprocessing errors.

    Raised when:
    - Invalid image format
    - Resize failure
    - Color conversion error
    """

    @classmethod
    def invalid_image(cls, shape: tuple) -> "PreprocessError":
        """Create error for invalid image dimensions."""
        return cls(
            f"Invalid image shape: {shape}",
            details={"shape": shape},
            suggestions=[
                "Expected 3-channel (RGB/BGR) image",
                "Check video source is providing valid frames",
            ],
        )


class OutputError(HunterError):
    """
    Output sink errors.

    Raised when:
    - Failed to write output
    - Invalid output format
    - Connection errors (for network sinks)
    """

    @classmethod
    def write_failed(cls, path: Path, error: str) -> "OutputError":
        """Create error for output write failure."""
        return cls(
            f"Failed to write output to: {path}",
            details={"path": str(path), "error": error},
            suggestions=[
                f"Check write permissions: ls -la {path.parent}",
                "Ensure parent directory exists",
                "Check disk space: df -h",
            ],
        )

    @classmethod
    def connection_failed(cls, endpoint: str, error: str) -> "OutputError":
        """Create error for network output failure."""
        return cls(
            f"Failed to connect to output endpoint: {endpoint}",
            details={"endpoint": endpoint, "error": error},
            suggestions=[
                "Check network connectivity",
                "Verify endpoint is running and accessible",
                "Check firewall settings",
            ],
        )
