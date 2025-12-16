# Hunter Drone Detection & Tracking System
# Multi-stage Dockerfile for production deployment
#
# Build: docker build -t hunter-drone:latest .
# Run:   docker run --gpus all -v ./data:/app/data hunter-drone:latest

# ============================================
# Stage 1: Base image with CUDA and Python
# ============================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Build tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================
# Stage 2: Dependencies installation
# ============================================
FROM base AS dependencies

WORKDIR /app

# Copy only dependency files first (for better caching)
COPY requirements.txt .
COPY pyproject.toml .

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 3: Development image
# ============================================
FROM dependencies AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-benchmark>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    ruff>=0.1.0 \
    mypy>=1.5.0

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev]"

# Set default command for development
CMD ["pytest", "tests/", "-v"]

# ============================================
# Stage 4: Production image
# ============================================
FROM dependencies AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 hunter \
    && useradd --uid 1000 --gid hunter --shell /bin/bash --create-home hunter

# Copy source code
COPY --chown=hunter:hunter src/ src/
COPY --chown=hunter:hunter configs/ configs/
COPY --chown=hunter:hunter scripts/ scripts/
COPY --chown=hunter:hunter pyproject.toml .
COPY --chown=hunter:hunter README.md .

# Install package
RUN pip install --no-cache-dir .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/outputs /app/logs \
    && chown -R hunter:hunter /app

# Switch to non-root user
USER hunter

# Environment variables for runtime
ENV HUNTER_CONFIG_PATH=/app/configs/default.yaml
ENV HUNTER_MODEL_PATH=/app/models
ENV HUNTER_LOG_PATH=/app/logs
ENV HUNTER_OUTPUT_PATH=/app/outputs

# Expose port for potential API service
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from hunter import Pipeline; print('OK')" || exit 1

# Default command
CMD ["hunter-run", "--config", "/app/configs/default.yaml"]

# ============================================
# Stage 5: Training image
# ============================================
FROM dependencies AS training

WORKDIR /app

# Install training dependencies
RUN pip install --no-cache-dir \
    tensorboard>=2.14.0 \
    mlflow>=2.7.0 \
    albumentations>=1.3.0 \
    wandb>=0.15.0

# Copy source code
COPY . .

# Install package with training extras
RUN pip install -e ".[training]"

# Create directories
RUN mkdir -p /app/data /app/models /app/experiments /app/logs

# Expose TensorBoard port
EXPOSE 6006

# Default command for training
CMD ["python", "scripts/train.py"]
