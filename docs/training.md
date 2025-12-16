# Training Guide

This guide covers how to train custom drone detection models using the Hunter Training Studio GUI.

## Table of Contents

- [Overview](#overview)
- [Quick Start with GUI](#quick-start-with-gui)
- [Training Studio Interface](#training-studio-interface)
- [Dataset Wizard](#dataset-wizard)
- [One-Click Training](#one-click-training)
- [Live Monitoring Dashboard](#live-monitoring-dashboard)
- [Auto-Optimization](#auto-optimization)
- [Model Export Wizard](#model-export-wizard)
- [Advanced: Command Line](#advanced-command-line)

## Overview

Hunter Drone provides a **Training Studio** GUI that automates the entire training process. No command-line knowledge required - everything is handled through an intuitive interface.

### What Gets Automated

| Task | Manual Approach | Hunter Studio |
|------|-----------------|---------------|
| Dataset organization | Manual folder structure | Drag & drop |
| Label format conversion | Python scripts | Automatic |
| Train/val split | Manual copying | One click |
| Hyperparameter tuning | Trial and error | Auto-optimized |
| Training monitoring | TensorBoard setup | Built-in dashboard |
| Issue detection | Manual log analysis | Real-time alerts |
| Model export | Multiple commands | Export wizard |

## Quick Start with GUI

### Launch Training Studio

```bash
hunter-studio
```

Or from Python:

```python
from hunter.studio import launch_training_studio

launch_training_studio()
```

The Training Studio opens in your default browser at `http://localhost:8501`.

### 3-Step Training Process

```
┌─────────────────────────────────────────────────────────────┐
│                   HUNTER TRAINING STUDIO                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐            │
│   │  STEP 1 │ ──── │  STEP 2 │ ──── │  STEP 3 │            │
│   │ Dataset │      │  Train  │      │ Deploy  │            │
│   └─────────┘      └─────────┘      └─────────┘            │
│                                                             │
│   Import your      Click Start      Export optimized       │
│   images & labels  Training         model                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training Studio Interface

### Main Dashboard

When you open Training Studio, you see:

```
┌──────────────────────────────────────────────────────────────┐
│  HUNTER TRAINING STUDIO                          [Settings]  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │                │  │                │  │                │ │
│  │  New Project   │  │  Open Project  │  │  Recent        │ │
│  │      [+]       │  │      [📁]      │  │  Projects      │ │
│  │                │  │                │  │                │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  Quick Actions:                                              │
│  [🎯 Start Training]  [📊 View Results]  [📦 Export Model]  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Creating a New Project

1. Click **New Project**
2. Enter project name (e.g., "DroneDetector_v1")
3. Select project folder
4. Click **Create**

The system automatically creates the required folder structure.

## Dataset Wizard

### Importing Images

The Dataset Wizard handles all data preparation automatically.

#### Option 1: Drag & Drop

Simply drag your image folder into the import area:

```
┌──────────────────────────────────────────────────────────────┐
│  DATASET WIZARD - Step 1: Import Images                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                        │ │
│  │     📁 Drag & Drop your image folder here             │ │
│  │                                                        │ │
│  │     or click to browse                                │ │
│  │                                                        │ │
│  │     Supported: JPG, PNG, BMP, MP4, AVI                │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  [Extract frames from video]  [Import labeled dataset]       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Option 2: Video Frame Extraction

For video files, the wizard automatically extracts frames:

1. Drop video file
2. Set frame interval (default: every 30 frames)
3. Click **Extract**

```
Extracting frames from: drone_footage.mp4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 1,234 frames extracted
```

### Labeling Interface

Built-in labeling tool for annotating drones:

```
┌──────────────────────────────────────────────────────────────┐
│  LABELING TOOL                           Image 45 of 1,234   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────┐  ┌──────────────────┐ │
│  │                                  │  │ Classes:         │ │
│  │     [Image with drone]           │  │ ☑ drone          │ │
│  │          ┌─────┐                 │  │ ☐ bird           │ │
│  │          │ 🎯  │ ← bbox          │  │ ☐ plane          │ │
│  │          └─────┘                 │  │                  │ │
│  │                                  │  │ [+ Add Class]    │ │
│  │                                  │  ├──────────────────┤ │
│  │                                  │  │ This Image:      │ │
│  │                                  │  │ • drone (0.95)   │ │
│  └──────────────────────────────────┘  └──────────────────┘ │
│                                                              │
│  Tools: [□ Box] [✎ Edit] [🗑 Delete]    [← Prev] [Next →]   │
│                                                              │
│  Shortcuts: B=Box, D=Delete, Space=Next, Backspace=Prev     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Auto-Split Dataset

The wizard automatically splits your data:

```
┌──────────────────────────────────────────────────────────────┐
│  DATASET WIZARD - Step 3: Split Dataset                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Total Images: 1,234                                         │
│  Total Annotations: 2,456 drones                             │
│                                                              │
│  Split Ratio:                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Train: 70%  ████████████████████░░░░░░░░  864 imgs  │    │
│  │ Val:   20%  ██████░░░░░░░░░░░░░░░░░░░░░░  247 imgs  │    │
│  │ Test:  10%  ███░░░░░░░░░░░░░░░░░░░░░░░░░  123 imgs  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ☑ Stratified split (maintain class balance)                │
│  ☑ Shuffle before split                                     │
│                                                              │
│                              [Apply Split & Continue →]      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## One-Click Training

### Training Configuration

Select a preset or let the system auto-configure:

```
┌──────────────────────────────────────────────────────────────┐
│  TRAINING SETUP                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Preset:                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ◉ Balanced (Recommended)                               │ │
│  │   YOLO11m | ~30 FPS | Good accuracy                    │ │
│  │                                                        │ │
│  │ ○ Fast                                                 │ │
│  │   YOLO11n | ~80 FPS | For real-time                    │ │
│  │                                                        │ │
│  │ ○ Accurate                                             │ │
│  │   YOLO11x | ~15 FPS | Maximum accuracy                 │ │
│  │                                                        │ │
│  │ ○ Custom...                                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Hardware: NVIDIA RTX 3080 (10GB) ✓ Detected                │
│                                                              │
│  Estimated Training Time: ~2 hours                          │
│                                                              │
│                    [🚀 Start Training]                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### What Happens Automatically

When you click **Start Training**, the system:

1. **Validates dataset** - Checks for errors, missing labels
2. **Optimizes hyperparameters** - Based on your GPU and dataset size
3. **Configures augmentation** - Appropriate for drone detection
4. **Sets up monitoring** - Real-time metrics dashboard
5. **Enables auto-save** - Checkpoints every 10 epochs
6. **Activates Training Advisor** - Detects issues in real-time

## Live Monitoring Dashboard

### Real-Time Training View

```
┌──────────────────────────────────────────────────────────────┐
│  TRAINING IN PROGRESS                    Epoch 45/100        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Progress: ━━━━━━━━━━━━━━━━━━━░░░░░░░░░░░░  45%             │
│  ETA: 1h 15m remaining                                       │
│                                                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │ Loss                    │  │ mAP50                   │   │
│  │    ╭─╮                  │  │                    ╭──  │   │
│  │   ╱  ╰─╮                │  │               ╭───╯     │   │
│  │  ╱     ╰──╮             │  │          ╭───╯         │   │
│  │ ╱         ╰───────      │  │     ╭───╯              │   │
│  │╱                        │  │ ───╯                   │   │
│  └─────────────────────────┘  └─────────────────────────┘   │
│   Train: 0.023  Val: 0.031      Current: 0.89  Best: 0.91   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Training Advisor:                                    │    │
│  │ ✓ Training is progressing normally                   │    │
│  │ ✓ No overfitting detected                            │    │
│  │ ℹ Best model saved at epoch 42                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  [⏸ Pause]  [⏹ Stop]  [📊 Detailed Metrics]                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Automatic Issue Alerts

The Training Advisor monitors and alerts you:

```
┌──────────────────────────────────────────────────────────────┐
│  ⚠️ TRAINING ADVISOR ALERT                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Issue Detected: Possible Overfitting                        │
│                                                              │
│  Details:                                                    │
│  • Validation loss increasing for 5 consecutive epochs       │
│  • Train/Val gap: 0.15 (threshold: 0.10)                    │
│                                                              │
│  Recommended Actions:                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ◉ Apply Early Stopping (Recommended)                   │ │
│  │   Stop training and use best checkpoint                │ │
│  │                                                        │ │
│  │ ○ Increase Augmentation                                │ │
│  │   Add more data augmentation and continue              │ │
│  │                                                        │ │
│  │ ○ Ignore and Continue                                  │ │
│  │   Continue training without changes                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│                         [Apply Fix]  [Dismiss]               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Auto-Optimization

### Hyperparameter Auto-Tuning

Enable auto-tuning to find optimal settings:

```
┌──────────────────────────────────────────────────────────────┐
│  AUTO-OPTIMIZATION                                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ☑ Enable Auto-Optimization                                  │
│                                                              │
│  The system will automatically tune:                         │
│  • Learning rate (finds optimal starting rate)               │
│  • Batch size (maximizes GPU utilization)                    │
│  • Augmentation strength (balances variety vs quality)       │
│  • Early stopping patience                                   │
│                                                              │
│  Optimization Strategy:                                      │
│  ◉ Quick (5 trial runs) - ~30 min                           │
│  ○ Thorough (20 trial runs) - ~2 hours                      │
│  ○ Exhaustive (50 trial runs) - ~5 hours                    │
│                                                              │
│  Found Parameters:                                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ learning_rate: 0.0012 (auto-detected)                  │ │
│  │ batch_size: 24 (optimized for your GPU)                │ │
│  │ augmentation: medium                                   │ │
│  │ patience: 25 epochs                                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Learning Rate Finder

Automatic learning rate discovery:

```
Learning Rate Finder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

Suggested LR: 0.0012
        │
    ────┼────────────────
        │       ╭╮
        │      ╱  ╲
        │     ╱    ╲
        │────╯      ╰────
        └────────────────→
         10⁻⁵    10⁻³   10⁻¹

✓ Learning rate automatically configured
```

## Model Export Wizard

### One-Click Export

After training completes:

```
┌──────────────────────────────────────────────────────────────┐
│  EXPORT MODEL                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Training Complete! ✓                                        │
│  Best Model: epoch_87 (mAP50: 0.923)                        │
│                                                              │
│  Export Format:                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ☑ PyTorch (.pt) - For Hunter Pipeline                  │ │
│  │ ☐ ONNX (.onnx) - Cross-platform                        │ │
│  │ ☐ TensorRT (.engine) - NVIDIA optimized                │ │
│  │ ☐ CoreML (.mlmodel) - Apple devices                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Optimization:                                               │
│  ☑ FP16 (half precision) - 2x faster, minimal accuracy loss │
│                                                              │
│  Export Location: models/drone_detector_v1.pt               │
│                                                              │
│                           [📦 Export Model]                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Auto-Integration

The exported model is automatically configured for use:

```
✓ Model exported to: models/drone_detector_v1.pt
✓ Config updated: configs/default.yaml
✓ Ready to use with: hunter-run --source video.mp4

Quick Test:
┌────────────────────────────────────────────────────────────┐
│ [▶ Test on Sample Video]  [▶ Test on Webcam]              │
└────────────────────────────────────────────────────────────┘
```

## Training Profiles

### Pre-configured Profiles

| Profile | Model | Training Time | Use Case |
|---------|-------|---------------|----------|
| **Quick Test** | YOLO11n | ~15 min | Verify dataset |
| **Balanced** | YOLO11m | ~2 hours | General use |
| **High Accuracy** | YOLO11x | ~6 hours | Maximum precision |
| **Edge Device** | YOLO11n | ~30 min | Jetson/Raspberry Pi |

### Custom Profile

For advanced users, create custom profiles through the GUI:

```
┌──────────────────────────────────────────────────────────────┐
│  CUSTOM TRAINING PROFILE                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Model:        [YOLO11m ▼]                                  │
│  Epochs:       [100    ]                                    │
│  Batch Size:   [Auto ▼ ]  (or specify)                      │
│  Image Size:   [640 ▼  ]                                    │
│                                                              │
│  Advanced (auto-configured if left empty):                   │
│  Learning Rate: [      ]  Suggested: 0.001                  │
│  Optimizer:     [AdamW ▼]                                   │
│  Augmentation:  [Medium ▼]                                  │
│                                                              │
│  [Save as Profile]  [Reset to Defaults]                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Advanced: Command Line

For users who prefer command-line interface, all GUI functionality is also available via CLI.

### CLI Training

```bash
# Quick training with auto-configuration
hunter-train --data datasets/drones --preset balanced

# Custom configuration
hunter-train \
    --data datasets/drones \
    --model yolo11m \
    --epochs 100 \
    --auto-tune
```

### CLI Dataset Preparation

```bash
# Extract frames from video
hunter-dataset extract --video footage.mp4 --interval 30

# Auto-split dataset
hunter-dataset split --path datasets/drones --ratio 70:20:10

# Validate dataset
hunter-dataset validate --path datasets/drones
```

### CLI Export

```bash
# Export with optimization
hunter-export \
    --model runs/train/best.pt \
    --format onnx \
    --fp16
```

### Manual Configuration

For complete manual control, see:
- [Configuration Reference](configuration.md) - All configuration options
- [API Reference](api-reference.md) - Python API for custom integrations

## Troubleshooting

### Common Issues (Auto-Resolved)

| Issue | GUI Resolution |
|-------|----------------|
| Out of memory | Automatically reduces batch size |
| Slow training | Suggests FP16 or smaller model |
| Poor accuracy | Recommends more data or augmentation |
| Overfitting | Applies early stopping |

### Getting Help

In the GUI, click **Help** → **Training Assistant** for context-aware help.

## Next Steps

- [User Guide](user-guide.md) - Running detection
- [Configuration Reference](configuration.md) - All configuration options
- [API Reference](api-reference.md) - Full API documentation
