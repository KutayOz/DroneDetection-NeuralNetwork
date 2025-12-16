# Hunter Drone Detection & Tracking System

Real-time drone detection and tracking system using YOLO11 and Siamese networks with hybrid routing architecture.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-419%20passing-brightgreen)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Training Your Own Model](#training-your-own-model)
- [Configuration](#configuration)
- [Training Advisor](#training-advisor)
- [Output Format](#output-format)
- [Performance](#performance)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

Hunter Drone is a production-ready computer vision system designed to detect and track drones in real-time video streams. It combines the speed of YOLO11 object detection with the precision of Siamese networks for robust tracking.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID DETECTION PIPELINE                        │
│                                                                         │
│  Frame → YOLO11 (Primary) ──┬── High Conf (>0.8) ──→ Track Directly    │
│                             │                                           │
│                             ├── Medium Conf (0.5-0.8) → Siamese ──→ Track│
│                             │                                           │
│                             └── Low Conf (<0.5) ──→ Discard            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Hybrid Detection Pipeline**: YOLO11 primary detector + Siamese secondary verifier
- **Multi-Object Tracking**: Kalman filter based tracking with Hungarian algorithm association
- **Eagle State Machine**: Robust track lifecycle (SEARCH → LOCK → TRACK → LOST → RECOVER)
- **Multiple Input Sources**: Video files, RTSP streams, webcams, GStreamer pipelines
- **Training Advisor**: Automatic training analysis and recommendations
- **Production Ready**: 419 tests, comprehensive configuration, performance monitoring

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/hunter-drone/hunter-drone.git
cd hunter-drone
python3.10 -m venv venv && source venv/bin/activate
pip install -e .

# 2. Download YOLO11 model
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
mv yolo11m.pt models/

# 3. Run on a video file
hunter-run --config configs/default.yaml --source video.mp4
```

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10.x | 3.10.12 |
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |
| VRAM | 4GB | 8GB+ |
| RAM | 8GB | 16GB+ |
| CUDA | 11.7+ | 12.0+ |

### Step-by-Step Installation

#### 1. Install Python 3.10

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# macOS (with Homebrew)
brew install python@3.10

# Windows: Download from python.org
```

#### 2. Install CUDA (for GPU support)

```bash
# Ubuntu - Install NVIDIA drivers and CUDA
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version
```

#### 3. Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/hunter-drone/hunter-drone.git
cd hunter-drone

# Create virtual environment (IMPORTANT: use Python 3.10)
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install hunter-drone
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"

# For training (includes tensorboard, mlflow)
pip install -e ".[training]"
```

#### 4. Download Models

```bash
# Create models directory
mkdir -p models

# Download YOLO11 medium model (recommended starting point)
python -c "from ultralytics import YOLO; model = YOLO('yolo11m.pt')"
mv yolo11m.pt models/

# Available models:
# - yolo11n.pt  (fastest, lower accuracy)
# - yolo11s.pt  (fast)
# - yolo11m.pt  (balanced) ← recommended
# - yolo11l.pt  (accurate)
# - yolo11x.pt  (most accurate, slowest)
```

#### 5. Verify Installation

```bash
# Run tests
pytest tests/ -v --tb=short

# Check GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Running Detection on Video Files

```bash
# Basic usage
hunter-run --config configs/default.yaml --source path/to/video.mp4

# With custom output
hunter-run --config configs/default.yaml \
    --source input.mp4 \
    --output results.jsonl

# Low latency mode (for real-time)
hunter-run --config configs/profiles/low_latency.yaml --source video.mp4

# High accuracy mode (for offline processing)
hunter-run --config configs/profiles/high_accuracy.yaml --source video.mp4
```

### Running Detection on RTSP Streams

```bash
# IP Camera
hunter-run --config configs/default.yaml \
    --source "rtsp://username:password@192.168.1.100:554/stream"

# Drone feed
hunter-run --config configs/default.yaml \
    --source "rtsp://drone_ip:8554/live"
```

### Running Detection on Webcam

```bash
# Default webcam (device 0)
hunter-run --config configs/default.yaml --source 0

# Specific webcam
hunter-run --config configs/default.yaml --source 2
```

### Python API

```python
from hunter import Pipeline, HunterConfig

# Load configuration
config = HunterConfig.from_yaml("configs/default.yaml")
config.ingest.source_uri = "video.mp4"
config.detector.model_path = "models/yolo11m.pt"

# Run detection pipeline
with Pipeline(config) as pipeline:
    for message in pipeline.run():
        print(f"Frame {message.frame_id}: {message.track_count} tracks")

        for track in message.tracks:
            print(f"  Track {track.track_id}: {track.state}")
            print(f"    Bbox: {track.bbox_xyxy}")
            print(f"    Confidence: {track.confidence:.2f}")
            print(f"    Velocity: {track.velocity_px_per_s}")
```

### Processing Results

```python
import json

# Read results file
with open("results.jsonl", "r") as f:
    for line in f:
        frame = json.loads(line)

        print(f"Frame {frame['frame_id']}:")
        print(f"  Tracks: {frame['track_count']}")
        print(f"  Latency: {frame['pipeline_metrics']['total_e2e_ms']:.1f}ms")

        for track in frame['tracks']:
            if track['state'] == 'TRACK':
                x1, y1, x2, y2 = track['bbox_xyxy']
                print(f"  Drone at ({x1}, {y1}) to ({x2}, {y2})")
```

## Training Your Own Model

### 1. Prepare Dataset

Organize your drone images in YOLO format:

```
database/
├── images/
│   ├── train/
│   │   ├── drone_001.jpg
│   │   ├── drone_002.jpg
│   │   └── ...
│   └── val/
│       ├── drone_100.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── drone_001.txt    # YOLO format: class x_center y_center width height
│   │   └── ...
│   └── val/
│       └── ...
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

### 2. Label Format

Each `.txt` file contains one line per object:
```
# class x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.2 0.15
0 0.3 0.7 0.1 0.08
```

### 3. Train the Model

```bash
# Basic training
python -c "
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolo11m.pt')

# Train on your dataset
results = model.train(
    data='database/drone_dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,  # GPU 0
    patience=20,  # Early stopping
    save=True,
    project='runs/train',
    name='drone_detector'
)
"

# Copy best model
cp runs/train/drone_detector/weights/best.pt models/drone_yolo11m.pt
```

### 4. Training Tips

| Dataset Size | Epochs | Batch Size | Notes |
|--------------|--------|------------|-------|
| < 1,000 images | 200-300 | 8-16 | Use heavy augmentation |
| 1,000 - 10,000 | 100-150 | 16-32 | Standard training |
| > 10,000 | 50-100 | 32-64 | May need less epochs |

Recommended augmentation settings:
```python
results = model.train(
    data='database/drone_dataset.yaml',
    epochs=100,
    batch=16,
    # Augmentation
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)
```

### 5. Analyze Training with Training Advisor

```python
from hunter.training_advisor import TrainingAdvisor

advisor = TrainingAdvisor()

# Analyze training results
report = advisor.analyze("runs/train/drone_detector")
print(report)

# Get detailed issues and recommendations
result = advisor.run_full_analysis("runs/train/drone_detector")
for issue in result['issues']:
    print(f"Issue: {issue}")
for rec in result['recommendations']:
    print(f"Recommendation: {rec}")
```

## Configuration

### Configuration Profiles

| Profile | Model | Use Case | Expected FPS |
|---------|-------|----------|--------------|
| `default.yaml` | YOLO11m | Balanced performance | 25-30 |
| `low_latency.yaml` | YOLO11n | Real-time applications | 50-120 |
| `high_accuracy.yaml` | YOLO11x | Offline processing | 10-15 |

### Key Configuration Options

```yaml
# configs/default.yaml

# ============================================
# Video Input
# ============================================
ingest:
  source_type: "file"      # file, rtsp, stub
  source_uri: ""           # Path to video or RTSP URL
  target_fps: 30.0         # Target processing FPS
  buffer_size: 5           # Frame buffer size

# ============================================
# Detection
# ============================================
detector:
  primary:
    model_path: "models/yolo11m.pt"
    confidence_threshold: 0.25  # Detection threshold
    nms_threshold: 0.45         # NMS threshold
    device: "cuda:0"            # cuda:0, cpu, mps
    half_precision: true        # FP16 inference

  # Hybrid routing thresholds
  routing:
    high_confidence_threshold: 0.8    # Direct to tracking
    medium_confidence_threshold: 0.5  # Verify with Siamese
    low_confidence_threshold: 0.25    # Discard

# ============================================
# Tracking
# ============================================
tracking:
  # State machine
  state_machine:
    lock_confirm_frames: 3      # Frames to confirm detection
    lost_timeout_frames: 30     # Frames before track is lost
    recover_max_frames: 15      # Max recovery attempts

  # Association
  association:
    iou_weight: 0.5            # IoU importance
    embedding_weight: 0.3      # Appearance importance
    motion_weight: 0.2         # Motion importance
    iou_threshold: 0.3         # Minimum IoU for match

# ============================================
# Output
# ============================================
output:
  sink_type: "json"            # json, stub, udp
  output_path: "outputs/tracks.jsonl"
  include_trajectory: true
  include_metrics: true
```

### Environment-Specific Settings

```bash
# Development
hunter-run --config configs/default.yaml --source video.mp4

# Production (with monitoring)
hunter-run --config configs/default.yaml \
    --source rtsp://camera:554/stream \
    --metrics-port 9090

# GPU Selection
CUDA_VISIBLE_DEVICES=1 hunter-run --config configs/default.yaml --source video.mp4
```

## Training Advisor

The Training Advisor analyzes your YOLO training runs and provides actionable recommendations.

### Basic Usage

```python
from hunter.training_advisor import TrainingAdvisor

# Create advisor
advisor = TrainingAdvisor()

# Analyze training results
report = advisor.analyze("runs/train/drone_detector")
print(report)
```

### Detected Issues

| Issue Type | Description | Auto-Fix |
|------------|-------------|----------|
| OVERFITTING | Val loss increasing while train loss decreases | Yes |
| LR_TOO_HIGH | Loss diverging or oscillating | Yes |
| LR_TOO_LOW | Very slow learning | Yes |
| PLATEAU | Loss not improving | Yes |
| LOW_MAP | Detection accuracy below threshold | No |
| PRECISION_RECALL_IMBALANCE | Model too conservative/aggressive | No |

### Example Output

```
============================================================
                Training Analysis Report
============================================================
Generated: 2024-01-15 10:30:45

[Summary]
  Issues found: 2
  Recommendations: 4

[Issues Detected]
  🔴 [HIGH] OVERFITTING (epoch 45): Train-val gap increasing
  ⚠️ [MEDIUM] PLATEAU (epoch 48): Training plateaued

[Recommendations]
  1. Enable early stopping with patience=10 [AUTO]
  2. Add weight decay regularization [AUTO]
  3. Increase data augmentation strength [AUTO]
  4. Consider using a smaller model variant
```

### Auto-Tuning

```python
from hunter.training_advisor import TrainingAdvisor

advisor = TrainingAdvisor()

# Get recommendations
result = advisor.run_full_analysis("runs/train/drone_detector")

# Auto-apply safe recommendations
tune_result = advisor.auto_tune(
    result['recommendations'],
    config_path="hyp.yaml",
    dry_run=False  # Set True to preview changes
)

print(f"Applied: {tune_result['applied']}")
print(f"Skipped: {tune_result['skipped']}")
```

## Output Format

### Track Message (JSONL)

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

### Track States

| State | Description |
|-------|-------------|
| `SEARCH` | Looking for new detections |
| `LOCK` | Detection found, confirming |
| `TRACK` | Actively tracking |
| `LOST` | Detection lost, predicting |
| `RECOVER` | Attempting to re-acquire |
| `DROP` | Track terminated |

## Performance

### Benchmarks (RTX 3080, 640x640)

| Model | FPS | Latency (p95) | mAP@50 |
|-------|-----|---------------|--------|
| YOLO11n | 120+ | 12ms | 0.85 |
| YOLO11s | 90+ | 15ms | 0.88 |
| YOLO11m | 60+ | 22ms | 0.91 |
| YOLO11l | 45+ | 32ms | 0.93 |
| YOLO11x | 30+ | 45ms | 0.95 |

### Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Latency p95 | ≤ 120ms | ≤ 200ms |
| Throughput | ≥ 25 FPS | ≥ 15 FPS |
| Detection F1 | ≥ 0.95 | ≥ 0.90 |
| ID Switch Rate | < 5% | < 10% |

### Optimization Tips

1. **Use FP16**: Enable `half_precision: true` for ~2x speedup
2. **Reduce resolution**: Use 416x416 instead of 640x640 for faster inference
3. **Skip frames**: Set `skip_frames: 1` to process every other frame
4. **Use TensorRT**: Export model to TensorRT for best GPU performance

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/hunter --cov-report=html

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/ -m "not slow" -v           # Skip slow tests
pytest tests/unit/test_tracking.py -v    # Specific module

# Run benchmarks
pytest tests/ -m benchmark --benchmark-only
```

## Project Structure

```
hunter_drone/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   └── profiles/              # Performance profiles
│       ├── low_latency.yaml
│       └── high_accuracy.yaml
├── database/                   # Your datasets (gitignored)
├── models/                     # Model weights (gitignored)
├── src/hunter/
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   ├── logger.py          # Structured logging
│   │   ├── metrics.py         # Performance metrics
│   │   └── timer.py           # Pipeline timing
│   ├── detection/             # Detection module
│   │   ├── yolo11_detector.py # YOLO11 implementation
│   │   ├── siamese_embedder.py# Siamese network
│   │   └── hybrid_detector.py # Hybrid routing
│   ├── tracking/              # Tracking module
│   │   ├── tracker.py         # Main tracker
│   │   ├── kalman_filter.py   # Kalman filter
│   │   ├── association.py     # Hungarian algorithm
│   │   ├── state_machine.py   # Eagle model
│   │   └── trajectory.py      # Trajectory management
│   ├── pipeline/              # Pipeline module
│   │   ├── ingest/            # Video input
│   │   ├── preprocess/        # Image preprocessing
│   │   ├── output/            # Output sinks
│   │   └── orchestrator.py    # Pipeline orchestration
│   ├── training_advisor/      # Training analysis
│   │   ├── collectors/        # Log parsers
│   │   ├── analyzers/         # Issue detectors
│   │   ├── engine/            # Decision engine
│   │   ├── reporters/         # Report generators
│   │   └── tuner/             # Auto-tuner
│   └── container/             # Dependency injection
├── tests/                     # Test suite (419 tests)
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── contract/              # Contract tests
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
# Or use smaller model (yolo11n instead of yolo11m)
# Or reduce input resolution to 416x416
```

**Low FPS**
```bash
# Enable half precision
# Use low_latency profile
# Skip frames if needed
```

**Poor Detection**
```bash
# Lower confidence_threshold
# Train on more diverse data
# Use larger model (yolo11l or yolo11x)
```

**Track ID Switches**
```bash
# Increase embedding_weight in association
# Increase lost_timeout_frames
# Use Siamese verification
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Citation

```bibtex
@software{hunter_drone,
  title = {Hunter Drone Detection & Tracking System},
  year = {2024},
  url = {https://github.com/hunter-drone/hunter-drone}
}
```
