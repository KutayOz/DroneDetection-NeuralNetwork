# Hunter Drone Detection & Tracking System

Real-time drone detection and multi-object tracking system using YOLO11 and Siamese networks.

## Features

- **YOLO11 Detection**: State-of-the-art object detection for drones
- **Siamese Embeddings**: Appearance-based re-identification
- **Kalman Filtering**: Smooth motion prediction
- **Hungarian Algorithm**: Optimal track-detection association
- **Eagle Model State Machine**: Robust track lifecycle management (SEARCH → LOCK → TRACK → LOST → RECOVER)

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## Installation

```bash
# Clone repository
cd hunter_drone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e .

# For development
pip install -e ".[dev]"

# For training
pip install -e ".[training]"
```

## Quick Start

### 1. Prepare Dataset

Place your drone dataset in the `database/` folder:

```
database/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── drone_dataset.yaml
```

Create `database/drone_dataset.yaml`:

```yaml
path: database
train: images/train
val: images/val

names:
  0: drone
```

### 2. Download/Train Model

**Option A: Use pretrained YOLO11**
```bash
# Download base model (will auto-download on first use)
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
mv yolo11m.pt models/
```

**Option B: Train on your dataset**
```bash
python scripts/run_training.py \
    --data database/drone_dataset.yaml \
    --model yolo11m.pt \
    --epochs 100 \
    --batch 16
```

### 3. Run Inference

```bash
# With config file
python scripts/run_inference.py \
    --config configs/default.yaml \
    --video path/to/video.mp4

# With output file
python scripts/run_inference.py \
    --config configs/default.yaml \
    --video input.mp4 \
    --output results.jsonl

# Low latency mode
python scripts/run_inference.py \
    --config configs/profiles/low_latency.yaml \
    --video input.mp4
```

### 4. Python API

```python
from hunter import Pipeline, HunterConfig

# Load config
config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"
config.detector.model_path = "models/yolo11m.pt"

# Run pipeline
with Pipeline(config) as pipeline:
    for message in pipeline.run():
        print(f"Frame {message.frame_id}: {message.track_count} tracks")

        for track in message.tracks:
            print(f"  Track {track.track_id}: {track.state} @ {track.bbox_xyxy}")
```

## Configuration

### Profiles

| Profile | Use Case | Model | FPS Target |
|---------|----------|-------|------------|
| `default.yaml` | Balanced | yolo11m | ~25 FPS |
| `low_latency.yaml` | Real-time | yolo11n | ~50+ FPS |
| `high_accuracy.yaml` | Offline | yolo11x | ~10 FPS |

### Key Parameters

```yaml
detector:
  confidence_threshold: 0.5  # Detection confidence
  nms_threshold: 0.45        # NMS IoU threshold

tracking:
  iou_threshold: 0.3         # Association IoU threshold
  embedding_weight: 0.3      # Appearance vs motion (0-1)
  lost_timeout_frames: 30    # Frames before track is lost
```

## Project Structure

```
hunter_drone/
├── configs/              # Configuration files
├── database/             # Your dataset (gitignored)
├── models/               # Model weights (gitignored)
├── scripts/              # Training/inference scripts
├── src/hunter/
│   ├── core/             # Config, logging, metrics
│   ├── models/           # YOLO11, Siamese
│   ├── tracking/         # Kalman, Hungarian, State Machine
│   ├── pipeline/         # Ingest, Preprocess, Output
│   └── utils/            # Bbox, IoU utilities
└── tests/                # Unit tests
```

## Output Format

```json
{
  "msg_version": "1.0",
  "timestamp_ms": 1702732800000,
  "frame_id": 100,
  "tracks": [
    {
      "track_id": 1,
      "state": "TRACK",
      "confidence": 0.95,
      "bbox_xyxy": [100, 100, 200, 200],
      "velocity_px_per_s": [50.0, -10.0]
    }
  ],
  "pipeline_metrics": {
    "detect_ms": 18.5,
    "total_e2e_ms": 35.2
  }
}
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Latency p95 | ≤ 120ms |
| Throughput | ≥ 25 FPS |
| Detection F1 | ≥ 0.95 |
| ID Switch Rate | < 5% |

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=hunter --cov-report=html

# Run specific test
pytest tests/unit/test_state_machine.py -v
```

## License

MIT License
