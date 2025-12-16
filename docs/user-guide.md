# User Guide

This guide covers how to use the Hunter Drone Detection & Tracking System for detecting and tracking drones in video streams.

## Table of Contents

- [Quick Start](#quick-start)
- [Input Sources](#input-sources)
- [Running Detection](#running-detection)
- [Understanding Output](#understanding-output)
- [Using the Python API](#using-the-python-api)
- [Performance Tuning](#performance-tuning)

## Quick Start

### Minimal Example

```bash
# Run detection on a video file
hunter-run --config configs/default.yaml --source video.mp4
```

### With Output File

```bash
# Save results to JSONL file
hunter-run --config configs/default.yaml \
    --source video.mp4 \
    --output results.jsonl
```

## Input Sources

Hunter Drone supports multiple input sources:

### 1. Video Files

Supported formats: MP4, AVI, MKV, MOV, and other formats supported by OpenCV.

```bash
# MP4 file
hunter-run --config configs/default.yaml --source /path/to/video.mp4

# AVI file
hunter-run --config configs/default.yaml --source recording.avi

# Relative path
hunter-run --config configs/default.yaml --source ./videos/test.mp4
```

### 2. RTSP Streams

Connect to IP cameras, drone feeds, or any RTSP-compatible device.

```bash
# Basic RTSP
hunter-run --config configs/default.yaml \
    --source "rtsp://192.168.1.100:554/stream"

# With authentication
hunter-run --config configs/default.yaml \
    --source "rtsp://admin:password@192.168.1.100:554/stream1"

# Different ports/paths
hunter-run --config configs/default.yaml \
    --source "rtsp://camera.local:8554/live"
```

**Common RTSP URL patterns:**

| Camera Brand | URL Pattern |
|--------------|-------------|
| Hikvision | `rtsp://admin:password@IP:554/Streaming/Channels/101` |
| Dahua | `rtsp://admin:password@IP:554/cam/realmonitor?channel=1` |
| Axis | `rtsp://IP:554/axis-media/media.amp` |
| Generic | `rtsp://IP:554/stream` |

### 3. Webcams

Use built-in or USB webcams.

```bash
# Default webcam (device 0)
hunter-run --config configs/default.yaml --source 0

# Second webcam (device 1)
hunter-run --config configs/default.yaml --source 1

# Specific device on Linux
hunter-run --config configs/default.yaml --source /dev/video0
```

### 4. Configuration-Based Source

Set the source in the configuration file:

```yaml
# configs/my_config.yaml
ingest:
  source_type: "file"  # or "rtsp"
  source_uri: "/path/to/video.mp4"
```

Then run:
```bash
hunter-run --config configs/my_config.yaml
```

## Running Detection

### Basic Commands

```bash
# Standard run
hunter-run --config configs/default.yaml --source video.mp4

# With output file
hunter-run --config configs/default.yaml --source video.mp4 --output results.jsonl

# Verbose mode
hunter-run --config configs/default.yaml --source video.mp4 --verbose
```

### Performance Profiles

Choose a profile based on your needs:

```bash
# Balanced (default) - 25-30 FPS, good accuracy
hunter-run --config configs/default.yaml --source video.mp4

# Low latency - 50+ FPS, for real-time applications
hunter-run --config configs/profiles/low_latency.yaml --source video.mp4

# High accuracy - 10-15 FPS, for detailed analysis
hunter-run --config configs/profiles/high_accuracy.yaml --source video.mp4
```

| Profile | Model | Target FPS | Use Case |
|---------|-------|------------|----------|
| Default | YOLO11m | 25-30 | General use |
| Low Latency | YOLO11n | 50-120 | Real-time tracking |
| High Accuracy | YOLO11x | 10-15 | Forensic analysis |

### GPU Selection

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 hunter-run --config configs/default.yaml --source video.mp4

# Use second GPU
CUDA_VISIBLE_DEVICES=1 hunter-run --config configs/default.yaml --source video.mp4

# Use CPU only
hunter-run --config configs/default.yaml --source video.mp4 --device cpu
```

## Understanding Output

### Console Output

During processing, you'll see:

```
[INFO] Starting Hunter Drone Detection System
[INFO] Loading model: models/yolo11m.pt
[INFO] Processing: video.mp4
[INFO] Frame 100: 2 tracks (active: 2, lost: 0)
[INFO] Frame 200: 3 tracks (active: 3, lost: 0)
[INFO] Frame 300: 2 tracks (active: 2, lost: 1)
...
[INFO] Processing complete
[INFO] Total frames: 1000
[INFO] Average FPS: 28.5
[INFO] Total tracks: 5
```

### JSONL Output Format

Each line in the output file is a JSON object representing one frame:

```json
{
  "msg_version": "1.0",
  "timestamp_ms": 1702732800000,
  "frame_id": 100,
  "track_count": 2,
  "tracks": [
    {
      "track_id": 1,
      "state": "TRACK",
      "confidence": 0.95,
      "bbox_xyxy": [100, 100, 200, 200],
      "predicted_bbox_xyxy": [105, 102, 205, 202],
      "velocity_px_per_s": [50.0, -10.0],
      "age_frames": 45,
      "hits": 43,
      "time_since_update": 0,
      "trajectory_tail": [
        {"t_ms": 1702732799000, "cx": 145.0, "cy": 148.0},
        {"t_ms": 1702732800000, "cx": 150.0, "cy": 150.0}
      ]
    }
  ],
  "model": {
    "detector_name": "YOLO11m",
    "detector_hash": "abc123"
  },
  "pipeline_metrics": {
    "detect_ms": 18.5,
    "embed_ms": 5.2,
    "track_ms": 2.1,
    "total_e2e_ms": 35.2
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `frame_id` | Sequential frame number |
| `track_id` | Unique identifier for each tracked object |
| `state` | Track state: SEARCH, LOCK, TRACK, LOST, RECOVER, DROP |
| `confidence` | Detection confidence (0.0 - 1.0) |
| `bbox_xyxy` | Bounding box [x1, y1, x2, y2] in pixels |
| `predicted_bbox_xyxy` | Kalman-predicted bounding box |
| `velocity_px_per_s` | Estimated velocity [vx, vy] in pixels/second |
| `age_frames` | Total frames this track has existed |
| `hits` | Number of successful detections |
| `time_since_update` | Frames since last detection |
| `trajectory_tail` | Recent position history |

### Track States

```
SEARCH → LOCK → TRACK ←→ LOST → DROP
                 ↑            ↓
                 └── RECOVER ─┘
```

| State | Description | Duration |
|-------|-------------|----------|
| `SEARCH` | Looking for detections | Initial |
| `LOCK` | Detection found, confirming | 3 frames |
| `TRACK` | Actively tracking | Indefinite |
| `LOST` | Detection lost, predicting | 30 frames max |
| `RECOVER` | Re-acquired after lost | 2 frames |
| `DROP` | Track terminated | Final |

## Using the Python API

### Basic Usage

```python
from hunter import Pipeline, HunterConfig

# Load configuration
config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

# Run pipeline
with Pipeline(config) as pipeline:
    for message in pipeline.run():
        print(f"Frame {message.frame_id}: {message.track_count} tracks")
```

### Processing Each Track

```python
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with Pipeline(config) as pipeline:
    for message in pipeline.run():
        for track in message.tracks:
            # Only process active tracks
            if track.state == "TRACK":
                x1, y1, x2, y2 = track.bbox_xyxy
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                print(f"Drone {track.track_id}:")
                print(f"  Position: ({center_x:.0f}, {center_y:.0f})")
                print(f"  Size: {width:.0f}x{height:.0f}")
                print(f"  Velocity: {track.velocity_px_per_s}")
                print(f"  Confidence: {track.confidence:.2f}")
```

### Custom Callbacks

```python
from hunter import Pipeline, HunterConfig

def on_new_track(track):
    """Called when a new track is confirmed."""
    print(f"New drone detected! Track ID: {track.track_id}")

def on_track_lost(track):
    """Called when a track is lost."""
    print(f"Lost track {track.track_id} after {track.age_frames} frames")

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with Pipeline(config) as pipeline:
    previous_tracks = set()

    for message in pipeline.run():
        current_tracks = {t.track_id for t in message.tracks if t.state == "TRACK"}

        # Detect new tracks
        for track in message.tracks:
            if track.track_id not in previous_tracks and track.state == "TRACK":
                on_new_track(track)

        # Detect lost tracks
        for track in message.tracks:
            if track.state == "DROP" and track.track_id in previous_tracks:
                on_track_lost(track)

        previous_tracks = current_tracks
```

### Saving Results to File

```python
import json
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

with open("results.jsonl", "w") as f:
    with Pipeline(config) as pipeline:
        for message in pipeline.run():
            # Convert to dict and save
            data = {
                "frame_id": message.frame_id,
                "timestamp_ms": message.timestamp_ms,
                "track_count": message.track_count,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "state": t.state,
                        "confidence": t.confidence,
                        "bbox_xyxy": t.bbox_xyxy,
                    }
                    for t in message.tracks
                ]
            }
            f.write(json.dumps(data) + "\n")
```

### Integration with OpenCV

```python
import cv2
from hunter import Pipeline, HunterConfig

config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"

# Open video for visualization
cap = cv2.VideoCapture("video.mp4")

with Pipeline(config) as pipeline:
    for message in pipeline.run():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw bounding boxes
        for track in message.tracks:
            if track.state == "TRACK":
                x1, y1, x2, y2 = map(int, track.bbox_xyxy)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"ID:{track.track_id} ({track.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display
        cv2.imshow("Hunter Drone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## Performance Tuning

### Quick Optimizations

1. **Use FP16 (Half Precision)**
   ```yaml
   detector:
     half_precision: true  # ~2x speedup
   ```

2. **Reduce Input Resolution**
   ```yaml
   preprocess:
     input_size: [416, 416]  # Instead of [640, 640]
   ```

3. **Skip Frames**
   ```yaml
   ingest:
     skip_frames: 1  # Process every other frame
   ```

4. **Use Smaller Model**
   ```yaml
   detector:
     model_path: "models/yolo11n.pt"  # Fastest model
   ```

### Accuracy vs Speed Trade-offs

| Setting | Speed Impact | Accuracy Impact |
|---------|--------------|-----------------|
| Smaller model | +++ faster | -- less accurate |
| Lower resolution | ++ faster | - slightly less accurate |
| Higher confidence threshold | + faster (fewer tracks) | +/- depends on scene |
| FP16 | ++ faster | minimal |
| Skip frames | ++ faster | - may miss fast objects |

### Recommended Settings by Use Case

**Real-time Monitoring (Security)**
```yaml
detector:
  model_path: "models/yolo11s.pt"
  half_precision: true
preprocess:
  input_size: [416, 416]
tracking:
  lost_timeout_frames: 15  # Quick cleanup
```

**Forensic Analysis (Accuracy)**
```yaml
detector:
  model_path: "models/yolo11x.pt"
  confidence_threshold: 0.4
preprocess:
  input_size: [1280, 1280]
tracking:
  lost_timeout_frames: 60  # Keep tracks longer
```

**Edge Device (Limited Resources)**
```yaml
detector:
  model_path: "models/yolo11n.pt"
  half_precision: true
preprocess:
  input_size: [320, 320]
ingest:
  skip_frames: 2
```

## Next Steps

- [Configuration Reference](configuration.md) - All configuration options
- [Training Guide](training.md) - Train custom drone detection models
- [API Reference](api-reference.md) - Full API documentation
