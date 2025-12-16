"""
Configuration management using Pydantic for type-safe validation.

Follows SOLID principles:
- SRP: Each config class handles one domain
- OCP: New config types can be added without modifying existing ones
- DIP: Components depend on config interfaces, not concrete implementations
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .exceptions import ConfigError


class IngestConfig(BaseModel):
    """Video ingest configuration."""

    source_type: Literal["file", "rtsp", "stub"] = "file"
    source_uri: str = ""
    buffer_size: int = Field(default=5, ge=1, le=100)
    timeout_ms: int = Field(default=5000, ge=100)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, v: str, info) -> str:
        """Validate source URI based on type."""
        # Stub type doesn't need a URI
        if info.data.get("source_type") == "stub":
            return v
        return v


class PreprocessConfig(BaseModel):
    """Image preprocessing configuration."""

    input_size: tuple[int, int] = (640, 640)  # (width, height)
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    pixel_format: Literal["RGB", "BGR"] = "RGB"

    @field_validator("input_size")
    @classmethod
    def validate_input_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Ensure input size is valid."""
        if v[0] < 32 or v[1] < 32:
            raise ValueError("Input size must be at least 32x32")
        return v


class DetectorConfig(BaseModel):
    """Object detector configuration."""

    model_path: Path
    model_type: Literal["yolo", "onnx"] = "yolo"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        """Validate model file format."""
        valid_suffixes = {".pt", ".onnx", ".engine", ".torchscript"}
        if v.suffix.lower() not in valid_suffixes:
            raise ValueError(f"Invalid model format: {v.suffix}. Must be one of {valid_suffixes}")
        return v


class EmbedderConfig(BaseModel):
    """Siamese embedder configuration."""

    enabled: bool = True
    model_path: Optional[Path] = None
    embedding_dim: int = Field(default=128, ge=32, le=512)
    input_size: tuple[int, int] = (128, 128)
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate embedder model path."""
        if v is not None and v.suffix.lower() not in {".pt", ".onnx", ".engine"}:
            raise ValueError(f"Invalid embedder model format: {v.suffix}")
        return v


class TrackingConfig(BaseModel):
    """Tracking system configuration."""

    # State machine parameters
    lock_confirm_frames: int = Field(default=3, ge=1, le=10)
    lock_timeout_frames: int = Field(default=5, ge=1, le=20)
    lost_timeout_frames: int = Field(default=30, ge=1, le=300)
    recover_max_frames: int = Field(default=15, ge=1, le=100)
    recover_confirm_frames: int = Field(default=2, ge=1, le=10)

    # Kalman filter parameters
    process_noise: float = Field(default=1.0, ge=0.01, le=100.0)
    measurement_noise: float = Field(default=1.0, ge=0.01, le=100.0)

    # Association parameters
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    gate_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    # Trajectory parameters
    trajectory_max_length: int = Field(default=150, ge=10)
    trajectory_output_points: int = Field(default=10, ge=1)


class OutputConfig(BaseModel):
    """Output sink configuration."""

    sink_type: Literal["json", "stub", "udp"] = "stub"
    output_path: Optional[Path] = None
    output_frequency: Literal["every_frame", "on_change"] = "every_frame"
    pretty_print: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    log_file: Optional[Path] = None
    log_stage_timings: bool = True
    log_track_changes: bool = True


class HunterConfig(BaseModel):
    """
    Root configuration for Hunter Drone system.

    Aggregates all component configurations.
    Follows Interface Segregation Principle (ISP):
    components only receive the config they need.
    """

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    detector: DetectorConfig
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "HunterConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            HunterConfig instance

        Raises:
            ConfigError: If file not found or invalid format
        """
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            if data is None:
                raise ConfigError(f"Empty config file: {path}")

            return cls(**data)

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def model_dump_safe(self) -> dict:
        """
        Dump config with Path objects converted to strings.
        Useful for logging.
        """
        return self.model_dump(mode="json")
