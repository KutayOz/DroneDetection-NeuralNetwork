# Configuration Reference

Complete reference for all Hunter Drone configuration options.

## Table of Contents

- [Configuration Files](#configuration-files)
- [Ingest Configuration](#ingest-configuration)
- [Preprocess Configuration](#preprocess-configuration)
- [Detector Configuration](#detector-configuration)
- [Embedder Configuration](#embedder-configuration)
- [Tracking Configuration](#tracking-configuration)
- [Output Configuration](#output-configuration)
- [Logging Configuration](#logging-configuration)
- [Configuration Profiles](#configuration-profiles)
- [Environment Variables](#environment-variables)

## Configuration Files

### File Format

Hunter Drone uses YAML configuration files:

```yaml
# configs/default.yaml
ingest:
  source_type: "file"
  source_uri: ""
  buffer_size: 5
  timeout_ms: 5000

preprocess:
  input_size: [640, 640]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"

detector:
  model_path: "models/yolo11m.pt"
  model_type: "yolo"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  max_detections: 100
  device: "cuda"

embedder:
  enabled: true
  model_path: null
  embedding_dim: 128
  input_size: [128, 128]
  device: "cuda"

tracking:
  lock_confirm_frames: 3
  lock_timeout_frames: 5
  lost_timeout_frames: 30
  recover_max_frames: 15
  recover_confirm_frames: 2
  process_noise: 1.0
  measurement_noise: 1.0
  iou_threshold: 0.3
  embedding_weight: 0.3
  gate_threshold: 0.1
  trajectory_max_length: 150
  trajectory_output_points: 10

output:
  sink_type: "json"
  output_path: null
  output_frequency: "every_frame"
  pretty_print: false

logging:
  level: "INFO"
  format: "json"
  log_file: null
  log_stage_timings: true
  log_track_changes: true
```

### Loading Configuration

```python
from hunter import HunterConfig
from pathlib import Path

# From YAML file
config = HunterConfig.from_yaml(Path("configs/default.yaml"))

# Programmatically
from hunter.core.config import (
    HunterConfig,
    IngestConfig,
    DetectorConfig,
)

config = HunterConfig(
    ingest=IngestConfig(source_uri="/path/to/video.mp4"),
    detector=DetectorConfig(model_path=Path("models/yolo11m.pt")),
)

# Save to YAML
config.to_yaml(Path("configs/my_config.yaml"))
```

## Ingest Configuration

Controls video input source.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `source_type` | string | `"file"` | `file`, `rtsp`, `stub` | Type of video source |
| `source_uri` | string | `""` | - | Path to video file or RTSP URL |
| `buffer_size` | int | `5` | 1-100 | Frame buffer size |
| `timeout_ms` | int | `5000` | >= 100 | Read timeout in milliseconds |

### Source Types

#### File Source
```yaml
ingest:
  source_type: "file"
  source_uri: "/path/to/video.mp4"
```

Supported formats: MP4, AVI, MKV, MOV, and other OpenCV-compatible formats.

#### RTSP Source
```yaml
ingest:
  source_type: "rtsp"
  source_uri: "rtsp://192.168.1.100:554/stream"
  buffer_size: 10
  timeout_ms: 10000
```

For unreliable networks, increase `buffer_size` and `timeout_ms`.

#### Stub Source (Testing)
```yaml
ingest:
  source_type: "stub"
  # source_uri not needed
```

Generates synthetic frames for testing.

## Preprocess Configuration

Controls image preprocessing before detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | [int, int] | `[640, 640]` | Input resolution [width, height] |
| `normalize_mean` | [float, float, float] | `[0.485, 0.456, 0.406]` | ImageNet mean normalization |
| `normalize_std` | [float, float, float] | `[0.229, 0.224, 0.225]` | ImageNet std normalization |
| `pixel_format` | string | `"RGB"` | `RGB` or `BGR` |

### Resolution Guidelines

| Resolution | Speed | Accuracy | Memory | Use Case |
|------------|-------|----------|--------|----------|
| `[320, 320]` | Fastest | Lower | ~1GB | Edge devices |
| `[416, 416]` | Fast | Good | ~2GB | Real-time |
| `[640, 640]` | Medium | Great | ~3GB | **Recommended** |
| `[1280, 1280]` | Slow | Best | ~6GB | High accuracy |

### Example
```yaml
preprocess:
  input_size: [640, 640]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  pixel_format: "RGB"
```

## Detector Configuration

Controls the YOLO object detector.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model_path` | Path | **required** | - | Path to model file |
| `model_type` | string | `"yolo"` | `yolo`, `onnx` | Model format |
| `confidence_threshold` | float | `0.5` | 0.0-1.0 | Minimum detection confidence |
| `nms_threshold` | float | `0.45` | 0.0-1.0 | NMS IoU threshold |
| `max_detections` | int | `100` | >= 1 | Maximum detections per frame |
| `device` | string | `"cuda"` | `cuda`, `cpu`, `mps` | Compute device |

### Model Types

| Model | Size | Speed (FPS) | mAP50 | Recommended For |
|-------|------|-------------|-------|-----------------|
| `yolo11n.pt` | 6MB | 80-120 | 0.75 | Edge, real-time |
| `yolo11s.pt` | 22MB | 50-80 | 0.82 | Balanced |
| `yolo11m.pt` | 52MB | 30-50 | 0.87 | **General use** |
| `yolo11l.pt` | 86MB | 20-35 | 0.90 | High accuracy |
| `yolo11x.pt` | 139MB | 10-20 | 0.92 | Offline analysis |

### Confidence Threshold

| Value | Effect |
|-------|--------|
| 0.3 | More detections, more false positives |
| 0.5 | Balanced (default) |
| 0.7 | Fewer detections, fewer false positives |
| 0.9 | Only high-confidence detections |

### Example
```yaml
detector:
  model_path: "models/yolo11m.pt"
  model_type: "yolo"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  max_detections: 100
  device: "cuda"
```

## Embedder Configuration

Controls the Siamese re-identification network.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | `true` | - | Enable/disable embedder |
| `model_path` | Path | `null` | - | Path to embedder model |
| `embedding_dim` | int | `128` | 32-512 | Embedding vector dimension |
| `input_size` | [int, int] | `[128, 128]` | - | Crop size for embedding |
| `device` | string | `"cuda"` | `cuda`, `cpu`, `mps` | Compute device |

### When to Enable/Disable

| Scenario | `enabled` | Notes |
|----------|-----------|-------|
| Standard tracking | `true` | Better identity preservation |
| Maximum speed | `false` | ~10% faster |
| Many similar objects | `true` | Reduces ID switches |
| Resource constrained | `false` | Less GPU memory |

### Example
```yaml
embedder:
  enabled: true
  model_path: "models/siamese_embedder.pt"
  embedding_dim: 128
  input_size: [128, 128]
  device: "cuda"
```

## Tracking Configuration

Controls the Eagle State Machine and Kalman filter tracking.

### State Machine Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `lock_confirm_frames` | int | `3` | 1-10 | Frames to confirm track (LOCK → TRACK) |
| `lock_timeout_frames` | int | `5` | 1-20 | Max frames in LOCK before DROP |
| `lost_timeout_frames` | int | `30` | 1-300 | Max frames in LOST before DROP |
| `recover_max_frames` | int | `15` | 1-100 | Max frames to attempt recovery |
| `recover_confirm_frames` | int | `2` | 1-10 | Frames to confirm recovery |

### Kalman Filter Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `process_noise` | float | `1.0` | 0.01-100.0 | Motion model uncertainty |
| `measurement_noise` | float | `1.0` | 0.01-100.0 | Detection measurement noise |

**Tuning guide:**
- High `process_noise`: Allows faster motion changes, more responsive
- High `measurement_noise`: Trusts predictions more, smoother tracks
- Low noise values: Tighter tracking, may lose fast-moving objects

### Association Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `iou_threshold` | float | `0.3` | 0.0-1.0 | Minimum IoU for detection-track matching |
| `embedding_weight` | float | `0.3` | 0.0-1.0 | Weight of embedding similarity in cost |
| `gate_threshold` | float | `0.1` | 0.0-1.0 | Gating threshold for associations |

### Trajectory Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `trajectory_max_length` | int | `150` | >= 10 | Maximum stored trajectory points |
| `trajectory_output_points` | int | `10` | >= 1 | Points included in output |

### Example Profiles

#### Fast Tracking (High FPS)
```yaml
tracking:
  lock_confirm_frames: 2
  lost_timeout_frames: 15
  process_noise: 2.0
  measurement_noise: 0.5
```

#### Robust Tracking (Occlusions)
```yaml
tracking:
  lock_confirm_frames: 4
  lost_timeout_frames: 60
  recover_max_frames: 30
  process_noise: 1.0
  measurement_noise: 2.0
```

#### Tight Tracking (Accurate Positions)
```yaml
tracking:
  lock_confirm_frames: 5
  process_noise: 0.5
  measurement_noise: 0.5
  iou_threshold: 0.5
```

## Output Configuration

Controls output format and destination.

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `sink_type` | string | `"stub"` | `json`, `stub`, `udp` | Output sink type |
| `output_path` | Path | `null` | - | File path for JSON output |
| `output_frequency` | string | `"every_frame"` | `every_frame`, `on_change` | When to output |
| `pretty_print` | bool | `false` | - | Format JSON with indentation |

### Sink Types

#### JSON File
```yaml
output:
  sink_type: "json"
  output_path: "results.jsonl"
  output_frequency: "every_frame"
  pretty_print: false
```

#### Stub (No Output)
```yaml
output:
  sink_type: "stub"
```

#### UDP Stream
```yaml
output:
  sink_type: "udp"
  # Additional UDP settings via environment
```

### Output Frequency

| Value | Description | Use Case |
|-------|-------------|----------|
| `every_frame` | Output for every processed frame | Complete logging |
| `on_change` | Output only when tracks change | Reduced file size |

## Logging Configuration

Controls logging behavior.

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `level` | string | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Log level |
| `format` | string | `"json"` | `json`, `console` | Log format |
| `log_file` | Path | `null` | - | File path for logs |
| `log_stage_timings` | bool | `true` | - | Log pipeline stage timings |
| `log_track_changes` | bool | `true` | - | Log track state changes |

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | All messages, including internal state |
| `INFO` | Standard operational messages |
| `WARNING` | Potential issues |
| `ERROR` | Errors only |

### Example
```yaml
logging:
  level: "INFO"
  format: "json"
  log_file: "logs/hunter.log"
  log_stage_timings: true
  log_track_changes: true
```

## Configuration Profiles

### Default (Balanced)

```yaml
# configs/profiles/default.yaml
detector:
  model_path: "models/yolo11m.pt"
  confidence_threshold: 0.5
  device: "cuda"

preprocess:
  input_size: [640, 640]

tracking:
  lock_confirm_frames: 3
  lost_timeout_frames: 30
```

### Low Latency

```yaml
# configs/profiles/low_latency.yaml
detector:
  model_path: "models/yolo11n.pt"
  confidence_threshold: 0.6
  device: "cuda"

preprocess:
  input_size: [416, 416]

tracking:
  lock_confirm_frames: 2
  lost_timeout_frames: 15

embedder:
  enabled: false
```

### High Accuracy

```yaml
# configs/profiles/high_accuracy.yaml
detector:
  model_path: "models/yolo11x.pt"
  confidence_threshold: 0.4
  device: "cuda"

preprocess:
  input_size: [1280, 1280]

tracking:
  lock_confirm_frames: 5
  lost_timeout_frames: 60

embedder:
  enabled: true
```

### CPU Only

```yaml
# configs/profiles/cpu.yaml
detector:
  model_path: "models/yolo11n.pt"
  device: "cpu"

preprocess:
  input_size: [320, 320]

embedder:
  enabled: false
  device: "cpu"
```

### Edge Device

```yaml
# configs/profiles/edge.yaml
detector:
  model_path: "models/yolo11n.pt"
  confidence_threshold: 0.6
  device: "cuda"

preprocess:
  input_size: [320, 320]

tracking:
  lost_timeout_frames: 10
  trajectory_max_length: 50

embedder:
  enabled: false
```

## Environment Variables

Override configuration via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU | `0`, `1`, `0,1` |
| `HUNTER_LOG_LEVEL` | Override log level | `DEBUG` |
| `HUNTER_MODEL_PATH` | Override model path | `/path/to/model.pt` |

### Usage

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 hunter-run --config configs/default.yaml --source video.mp4

# Enable debug logging
HUNTER_LOG_LEVEL=DEBUG hunter-run --config configs/default.yaml --source video.mp4
```

## Validation

Configuration is validated on load:

```python
from hunter import HunterConfig
from pydantic import ValidationError

try:
    config = HunterConfig.from_yaml("configs/invalid.yaml")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

Common validation errors:
- `confidence_threshold` must be 0.0-1.0
- `model_path` must have valid extension (.pt, .onnx, .engine)
- `input_size` must be at least 32x32

## Diger Dokumanlar

- [API Reference](api-reference.md) - Python API referansi
