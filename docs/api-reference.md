# API Reference

Complete Python API reference for the Hunter Drone Detection & Tracking System.

## Table of Contents

- [Pipeline](#pipeline)
- [Configuration](#configuration)
- [Messages](#messages)
- [Tracks](#tracks)
- [Ingest Module](#ingest-module)
- [Detector Module](#detector-module)
- [Tracker Module](#tracker-module)
- [Training Advisor](#training-advisor)

## Pipeline

The main entry point for running the detection system.

### `hunter.Pipeline`

```python
class Pipeline:
    """
    Main pipeline for drone detection and tracking.

    Coordinates all stages: ingest → preprocess → detect → embed → track → output
    """

    def __init__(self, config: HunterConfig) -> None:
        """
        Initialize the pipeline.

        Args:
            config: HunterConfig instance with all settings
        """

    def run(self) -> Generator[TrackingMessage, None, None]:
        """
        Run the pipeline, yielding messages for each frame.

        Yields:
            TrackingMessage for each processed frame

        Example:
            with Pipeline(config) as pipeline:
                for message in pipeline.run():
                    print(f"Frame {message.frame_id}: {message.track_count} tracks")
        """

    def __enter__(self) -> "Pipeline":
        """Context manager entry."""

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - releases resources."""
```

### Usage Examples

#### Basic Usage

```python
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with Pipeline(config) as pipeline:
    for message in pipeline.run():
        print(f"Frame {message.frame_id}: {message.track_count} tracks")
```

#### Processing Tracks

```python
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with Pipeline(config) as pipeline:
    for message in pipeline.run():
        for track in message.tracks:
            if track.state == "TRACK":
                x1, y1, x2, y2 = track.bbox_xyxy
                print(f"Drone {track.track_id} at ({x1}, {y1})")
```

#### Stopping Early

```python
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with Pipeline(config) as pipeline:
    for message in pipeline.run():
        if message.frame_id >= 100:
            break  # Stop after 100 frames
```

## Configuration

### `hunter.HunterConfig`

```python
class HunterConfig(BaseModel):
    """
    Root configuration for Hunter Drone system.

    Attributes:
        ingest: Video input configuration
        preprocess: Image preprocessing configuration
        detector: YOLO detector configuration
        embedder: Siamese embedder configuration
        tracking: Tracking system configuration
        output: Output sink configuration
        logging: Logging configuration
    """

    ingest: IngestConfig
    preprocess: PreprocessConfig
    detector: DetectorConfig
    embedder: EmbedderConfig
    tracking: TrackingConfig
    output: OutputConfig
    logging: LoggingConfig

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

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
```

### `hunter.core.config.IngestConfig`

```python
class IngestConfig(BaseModel):
    """Video ingest configuration."""

    source_type: Literal["file", "rtsp", "stub"] = "file"
    source_uri: str = ""
    buffer_size: int = 5  # Range: 1-100
    timeout_ms: int = 5000  # Range: >= 100
```

### `hunter.core.config.DetectorConfig`

```python
class DetectorConfig(BaseModel):
    """Object detector configuration."""

    model_path: Path  # Required
    model_type: Literal["yolo", "onnx"] = "yolo"
    confidence_threshold: float = 0.5  # Range: 0.0-1.0
    nms_threshold: float = 0.45  # Range: 0.0-1.0
    max_detections: int = 100  # Range: >= 1
    device: Literal["cuda", "cpu", "mps"] = "cuda"
```

### `hunter.core.config.TrackingConfig`

```python
class TrackingConfig(BaseModel):
    """Tracking system configuration."""

    # State machine
    lock_confirm_frames: int = 3  # Range: 1-10
    lock_timeout_frames: int = 5  # Range: 1-20
    lost_timeout_frames: int = 30  # Range: 1-300
    recover_max_frames: int = 15  # Range: 1-100
    recover_confirm_frames: int = 2  # Range: 1-10

    # Kalman filter
    process_noise: float = 1.0  # Range: 0.01-100.0
    measurement_noise: float = 1.0  # Range: 0.01-100.0

    # Association
    iou_threshold: float = 0.3  # Range: 0.0-1.0
    embedding_weight: float = 0.3  # Range: 0.0-1.0
    gate_threshold: float = 0.1  # Range: 0.0-1.0

    # Trajectory
    trajectory_max_length: int = 150  # Range: >= 10
    trajectory_output_points: int = 10  # Range: >= 1
```

## Messages

### `hunter.core.domain.TrackingMessage`

```python
@dataclass
class TrackingMessage:
    """
    Output message for each processed frame.

    Attributes:
        msg_version: Message format version
        timestamp_ms: Unix timestamp in milliseconds
        frame_id: Sequential frame number
        track_count: Number of tracks in this message
        tracks: List of track information
        model: Model information
        pipeline_metrics: Timing metrics
    """

    msg_version: str = "1.0"
    timestamp_ms: int
    frame_id: int
    track_count: int
    tracks: List[TrackInfo]
    model: ModelInfo
    pipeline_metrics: PipelineMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""

    def to_json(self) -> str:
        """Convert to JSON string."""
```

### `hunter.core.domain.PipelineMetrics`

```python
@dataclass
class PipelineMetrics:
    """
    Timing metrics for pipeline stages.

    Attributes:
        detect_ms: Detection time in milliseconds
        embed_ms: Embedding time in milliseconds
        track_ms: Tracking time in milliseconds
        total_e2e_ms: Total end-to-end time
    """

    detect_ms: float
    embed_ms: float
    track_ms: float
    total_e2e_ms: float
```

## Tracks

### `hunter.core.domain.TrackInfo`

```python
@dataclass
class TrackInfo:
    """
    Information about a tracked object.

    Attributes:
        track_id: Unique identifier for this track
        state: Current track state (SEARCH, LOCK, TRACK, LOST, RECOVER, DROP)
        confidence: Detection confidence (0.0-1.0)
        bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixels
        predicted_bbox_xyxy: Kalman-predicted bounding box
        velocity_px_per_s: Estimated velocity [vx, vy] in pixels/second
        age_frames: Total frames this track has existed
        hits: Number of successful detections
        time_since_update: Frames since last detection
        trajectory_tail: Recent position history
    """

    track_id: int
    state: str
    confidence: float
    bbox_xyxy: List[float]
    predicted_bbox_xyxy: List[float]
    velocity_px_per_s: List[float]
    age_frames: int
    hits: int
    time_since_update: int
    trajectory_tail: List[TrajectoryPoint]
```

### Track States

| State | Description | Transitions |
|-------|-------------|-------------|
| `SEARCH` | Looking for detections | → LOCK |
| `LOCK` | Detection found, confirming | → TRACK, DROP |
| `TRACK` | Actively tracking | → LOST |
| `LOST` | Detection lost, predicting | → RECOVER, DROP |
| `RECOVER` | Re-acquired after lost | → TRACK, DROP |
| `DROP` | Track terminated | Final |

### `hunter.core.domain.TrajectoryPoint`

```python
@dataclass
class TrajectoryPoint:
    """
    Single point in trajectory history.

    Attributes:
        t_ms: Timestamp in milliseconds
        cx: Center x coordinate
        cy: Center y coordinate
    """

    t_ms: int
    cx: float
    cy: float
```

## Ingest Module

### `hunter.ingest.IIngest`

```python
class IIngest(Protocol):
    """Interface for video sources."""

    def read(self) -> Optional[Frame]:
        """
        Read next frame from source.

        Returns:
            Frame object or None if no frame available
        """

    def release(self) -> None:
        """Release resources."""

    @property
    def fps(self) -> float:
        """Source frame rate."""

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Frame dimensions (width, height)."""
```

### `hunter.ingest.FileIngest`

```python
class FileIngest:
    """Video file input source."""

    def __init__(self, source_uri: str, config: IngestConfig) -> None:
        """
        Initialize file ingest.

        Args:
            source_uri: Path to video file
            config: Ingest configuration
        """
```

### `hunter.ingest.RTSPIngest`

```python
class RTSPIngest:
    """RTSP stream input source."""

    def __init__(self, source_uri: str, config: IngestConfig) -> None:
        """
        Initialize RTSP ingest.

        Args:
            source_uri: RTSP URL
            config: Ingest configuration
        """
```

## Detector Module

### `hunter.detector.IDetector`

```python
class IDetector(Protocol):
    """Interface for object detectors."""

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect objects in frame.

        Args:
            frame: Input frame

        Returns:
            List of Detection objects
        """

    @property
    def model_info(self) -> ModelInfo:
        """Model information."""
```

### `hunter.detector.YOLODetector`

```python
class YOLODetector:
    """YOLO-based object detector."""

    def __init__(self, config: DetectorConfig) -> None:
        """
        Initialize YOLO detector.

        Args:
            config: Detector configuration
        """

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Run YOLO detection on frame.

        Args:
            frame: Input frame

        Returns:
            List of Detection objects
        """
```

### `hunter.core.domain.Detection`

```python
@dataclass
class Detection:
    """
    Single detection from detector.

    Attributes:
        bbox_xyxy: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence (0.0-1.0)
        class_id: Class index
        class_name: Class name string
    """

    bbox_xyxy: List[float]
    confidence: float
    class_id: int
    class_name: str
```

## Tracker Module

### `hunter.tracker.ITracker`

```python
class ITracker(Protocol):
    """Interface for object trackers."""

    def update(
        self,
        detections: List[Detection],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections
            embeddings: Optional embeddings for re-identification

        Returns:
            List of current Track objects
        """

    def reset(self) -> None:
        """Reset tracker state."""
```

### `hunter.tracker.EagleTracker`

```python
class EagleTracker:
    """
    Eagle State Machine tracker.

    Implements the full tracking pipeline:
    - Kalman filter prediction
    - Hungarian algorithm association
    - State machine management
    """

    def __init__(self, config: TrackingConfig) -> None:
        """
        Initialize tracker.

        Args:
            config: Tracking configuration
        """

    def update(
        self,
        detections: List[Detection],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections
            embeddings: Optional embeddings

        Returns:
            List of current tracks
        """
```

### `hunter.tracker.Track`

```python
class Track:
    """
    Single tracked object.

    Attributes:
        track_id: Unique identifier
        state: Current state (TrackState enum)
        kalman_filter: KalmanFilter instance
        trajectory: Position history
        age: Total frames existed
        hits: Successful detection count
        time_since_update: Frames since last detection
    """

    @property
    def bbox(self) -> List[float]:
        """Current bounding box."""

    @property
    def velocity(self) -> List[float]:
        """Current velocity estimate."""

    @property
    def predicted_bbox(self) -> List[float]:
        """Predicted next bounding box."""
```

## Training Advisor

### `hunter.training_advisor.TrainingAdvisor`

```python
class TrainingAdvisor:
    """
    Main class for training analysis and recommendations.

    Analyzes training logs, detects issues, and provides
    actionable recommendations.
    """

    def __init__(self, config: Optional[AdvisorConfig] = None) -> None:
        """
        Initialize advisor.

        Args:
            config: Optional configuration
        """

    def analyze(
        self,
        source: Union[str, Path],
        source_type: str = "csv"
    ) -> str:
        """
        Run full analysis and return report.

        Args:
            source: Path to training logs
            source_type: Type of source ("csv", "yolo", "tensorboard")

        Returns:
            Formatted report string
        """

    def collect(
        self,
        source: Union[str, Path],
        source_type: str = "csv"
    ) -> TrainingMetrics:
        """
        Collect metrics from training logs.

        Args:
            source: Path to training logs
            source_type: Type of source

        Returns:
            TrainingMetrics instance
        """

    def detect_issues(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Detect training issues from metrics.

        Args:
            metrics: Training metrics

        Returns:
            List of detected issues
        """

    def recommend(self, issues: List[Issue]) -> List[Recommendation]:
        """
        Generate recommendations for issues.

        Args:
            issues: List of issues

        Returns:
            List of recommendations
        """

    def run_full_analysis(
        self,
        source: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.

        Args:
            source: Path to training logs

        Returns:
            Dictionary with metrics, issues, recommendations, reports
        """
```

### `hunter.training_advisor.Issue`

```python
@dataclass
class Issue:
    """
    Detected training issue.

    Attributes:
        issue_type: Type of issue (IssueType enum)
        severity: Issue severity (IssueSeverity enum)
        message: Human-readable description
        epoch_detected: Epoch where issue was detected
        details: Additional details dictionary
    """

    issue_type: IssueType
    severity: IssueSeverity
    message: str
    epoch_detected: int
    details: Dict[str, Any] = field(default_factory=dict)
```

### `hunter.training_advisor.IssueType`

```python
class IssueType(Enum):
    """Types of training issues."""

    OVERFITTING = "overfitting"
    LR_TOO_HIGH = "lr_too_high"
    LR_TOO_LOW = "lr_too_low"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    PLATEAU = "plateau"
    CLASS_IMBALANCE = "class_imbalance"
    DATA_QUALITY = "data_quality"
```

### `hunter.training_advisor.Recommendation`

```python
@dataclass
class Recommendation:
    """
    Training recommendation.

    Attributes:
        rec_type: Type of recommendation
        source_issue: Issue that triggered this recommendation
        message: Human-readable recommendation
        auto_applicable: Whether this can be auto-applied
        parameters: Suggested parameter values
    """

    rec_type: RecommendationType
    source_issue: IssueType
    message: str
    auto_applicable: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
```

### Usage Example

```python
from hunter.training_advisor import TrainingAdvisor

advisor = TrainingAdvisor()

# Full analysis
result = advisor.run_full_analysis("runs/detect/exp/results.csv")

# Print console report
print(result["reports"]["console"])

# Access individual components
for issue in result["issues"]:
    print(f"[{issue.severity.name}] {issue.issue_type.value}: {issue.message}")

for rec in result["recommendations"]:
    print(f"Recommendation: {rec.message}")
```

## Exceptions

### `hunter.core.exceptions.HunterError`

```python
class HunterError(Exception):
    """Base exception for Hunter Drone system."""
    pass
```

### `hunter.core.exceptions.ConfigError`

```python
class ConfigError(HunterError):
    """Configuration-related errors."""
    pass
```

### `hunter.core.exceptions.IngestError`

```python
class IngestError(HunterError):
    """Video ingest errors."""
    pass
```

### `hunter.core.exceptions.DetectorError`

```python
class DetectorError(HunterError):
    """Detection errors."""
    pass
```

### `hunter.core.exceptions.TrackerError`

```python
class TrackerError(HunterError):
    """Tracking errors."""
    pass
```

## Utilities

### `hunter.core.utils.compute_iou`

```python
def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value (0.0-1.0)
    """
```

### `hunter.core.utils.bbox_center`

```python
def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get center point of bounding box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Tuple of (center_x, center_y)
    """
```

## Next Steps

- [User Guide](user-guide.md) - Running detection
- [Training Guide](training.md) - Training custom models
- [Configuration Reference](configuration.md) - All configuration options
