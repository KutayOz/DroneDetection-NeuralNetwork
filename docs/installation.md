# Installation Guide

This guide provides detailed instructions for installing the Hunter Drone Detection & Tracking System.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installing Prerequisites](#installing-prerequisites)
- [Installing Hunter Drone](#installing-hunter-drone)
- [Downloading Models](#downloading-models)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 | Multi-core for parallel processing |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX 3080+ | Required for real-time performance |
| **VRAM** | 4GB | 8GB+ | More VRAM allows larger models |
| **RAM** | 8GB | 16GB+ | For video buffering |
| **Storage** | 10GB free | SSD recommended | For models and logs |

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| **Operating System** | Ubuntu 20.04+, Windows 10+, macOS 12+ | Linux recommended for production |
| **Python** | 3.10.x (required) | **Must be 3.10**, not 3.11 or 3.12 |
| **CUDA** | 11.7+ | For NVIDIA GPU support |
| **cuDNN** | 8.5+ | Deep learning acceleration |

### Why Python 3.10?

Hunter Drone requires Python 3.10 specifically due to:
- `filterpy` dependency (Kalman filtering) requires `numpy < 2.0`
- Type annotation compatibility
- Tested and validated configuration

## Installing Prerequisites

### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Install system dependencies
sudo apt install git build-essential libgl1-mesa-glx libglib2.0-0 -y

# Verify Python version
python3.10 --version  # Should show Python 3.10.x
```

### Installing NVIDIA Drivers and CUDA (Ubuntu)

```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-535 -y

# Reboot to load driver
sudo reboot

# After reboot, verify driver
nvidia-smi

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit -y

# Verify CUDA
nvcc --version
```

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10
brew install python@3.10

# Create alias (optional)
echo 'alias python3.10="/opt/homebrew/bin/python3.10"' >> ~/.zshrc
source ~/.zshrc
```

> **Note**: macOS with Apple Silicon (M1/M2/M3) uses MPS (Metal Performance Shaders) instead of CUDA.

### Windows

1. **Download Python 3.10** from [python.org](https://www.python.org/downloads/release/python-31012/)
   - Select "Windows installer (64-bit)"
   - During installation, check "Add Python to PATH"

2. **Install NVIDIA CUDA Toolkit**:
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select Windows → x86_64 → Your Windows version → exe (local)
   - Run installer and follow instructions

3. **Install Git**:
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default options during installation

4. **Verify Installation** (PowerShell):
   ```powershell
   python --version  # Should show Python 3.10.x
   nvidia-smi        # Should show GPU info
   ```

## Installing Hunter Drone

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/hunter-drone/hunter-drone.git
cd hunter-drone
```

### Step 2: Create Virtual Environment

```bash
# Linux/macOS
python3.10 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Hunter Drone

```bash
# Basic installation
pip install -e .

# With development tools (testing, linting)
pip install -e ".[dev]"

# With training tools (tensorboard, mlflow)
pip install -e ".[training]"

# Full installation
pip install -e ".[all]"
```

### Installation Options

| Option | Includes | Use Case |
|--------|----------|----------|
| `pip install -e .` | Core dependencies | Running inference only |
| `pip install -e ".[dev]"` | pytest, black, mypy | Development and testing |
| `pip install -e ".[training]"` | tensorboard, mlflow, wandb | Model training |
| `pip install -e ".[monitoring]"` | prometheus, opentelemetry | Production monitoring |
| `pip install -e ".[all]"` | Everything | Full development setup |

## Downloading Models

### Create Models Directory

```bash
mkdir -p models
```

### Option 1: Download Pre-trained YOLO11

```bash
# Download using Python
python -c "
from ultralytics import YOLO
# Download YOLO11 medium (recommended)
model = YOLO('yolo11m.pt')
"

# Move to models directory
mv yolo11m.pt models/
```

### Available YOLO11 Models

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| `yolo11n.pt` | 6MB | Fastest | Good | Edge devices, real-time |
| `yolo11s.pt` | 22MB | Fast | Better | Balanced deployments |
| `yolo11m.pt` | 52MB | Medium | Great | **General use (recommended)** |
| `yolo11l.pt` | 86MB | Slow | Excellent | High accuracy needs |
| `yolo11x.pt` | 139MB | Slowest | Best | Offline processing |

### Option 2: Use Custom Trained Model

If you have trained your own drone detection model:

```bash
# Copy your model to models directory
cp /path/to/your/best.pt models/drone_yolo11m.pt

# Update config to use your model
# In configs/default.yaml:
# detector:
#   model_path: "models/drone_yolo11m.pt"
```

## Verifying Installation

### Run Tests

```bash
# Run all tests (should show 419 passed)
pytest tests/ -v --tb=short

# Quick smoke test
pytest tests/unit/test_config.py -v
```

### Check GPU Detection

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

Expected output:
```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3080
GPU Memory: 10.0 GB
```

### Test Model Loading

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo11m.pt')
print(f'Model loaded successfully!')
print(f'Model type: {model.task}')
"
```

### Quick Inference Test

```bash
# Create a test with the stub ingest (no video needed)
python -c "
from hunter.core.config import HunterConfig, DetectorConfig
from pathlib import Path

# Create minimal config
config = HunterConfig(
    detector=DetectorConfig(model_path=Path('models/yolo11m.pt'))
)
print('Configuration loaded successfully!')
print(f'Detector model: {config.detector.model_path}')
"
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'hunter'"

**Cause**: Package not installed properly.

**Solution**:
```bash
# Make sure you're in the project directory
cd hunter_drone

# Reinstall in development mode
pip install -e .
```

#### "CUDA out of memory"

**Cause**: GPU doesn't have enough VRAM.

**Solutions**:
1. Use a smaller model:
   ```bash
   # In config, change to smaller model
   # model_path: "models/yolo11n.pt"
   ```

2. Reduce input resolution:
   ```yaml
   # In config
   preprocess:
     input_size: [416, 416]  # Instead of [640, 640]
   ```

3. Enable half precision:
   ```yaml
   detector:
     half_precision: true
   ```

#### "torch.cuda.is_available() returns False"

**Cause**: CUDA not properly installed or wrong PyTorch version.

**Solutions**:
1. Check NVIDIA driver:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

#### "Invalid model format" error

**Cause**: Model file is corrupted or wrong format.

**Solution**:
```bash
# Re-download the model
rm models/yolo11m.pt
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
mv yolo11m.pt models/
```

#### "Python version mismatch" errors

**Cause**: Using Python 3.11 or 3.12 instead of 3.10.

**Solution**:
```bash
# Check Python version
python --version

# Create new venv with correct Python
deactivate  # Exit current venv
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/hunter-drone/hunter-drone/issues)
2. Search for similar problems
3. Create a new issue with:
   - Your OS and version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Read the [User Guide](user-guide.md)** to learn how to run detection
2. **Check [Configuration Reference](configuration.md)** for customization options
3. **See [Training Guide](training.md)** if you want to train custom models
