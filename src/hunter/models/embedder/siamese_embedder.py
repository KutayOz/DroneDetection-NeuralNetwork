"""
Siamese network embedder implementation.

Generates appearance embeddings for:
- Track-detection association (reduces ID switches)
- Hard-negative rejection (distinguishing similar objects)
- Re-identification after occlusion

Uses ONNX Runtime for efficient inference.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from .base import BaseEmbedder
from ...core.config import EmbedderConfig
from ...core.exceptions import ModelError
from ...utils.checksum import compute_sha256

# Import ONNX Runtime with error handling
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ort = None


class SiameseEmbedder(BaseEmbedder):
    """
    Siamese network embedder using ONNX Runtime.

    Generates L2-normalized appearance embeddings from
    object crops for tracking association.

    Model Architecture:
    - Input: RGB image crop (H, W, 3)
    - Output: L2-normalized embedding vector

    Preprocessing:
    - Resize to configured input size
    - ImageNet normalization
    - HWC → CHW conversion
    """

    def __init__(self, config: EmbedderConfig):
        """
        Initialize Siamese embedder.

        Args:
            config: Embedder configuration

        Raises:
            ImportError: If onnxruntime is not installed
            ModelError: If model file not found or invalid
        """
        if not HAS_ONNX:
            raise ImportError(
                "onnxruntime package required for SiameseEmbedder. "
                "Install with: pip install onnxruntime or onnxruntime-gpu"
            )

        if config.model_path is None:
            raise ModelError("Embedder model_path not specified")

        self._config = config
        self._model_path = Path(config.model_path)

        # Validate model file
        if not self._model_path.exists():
            raise ModelError(f"Embedder model not found: {self._model_path}")

        # Compute model hash
        self._name = self._model_path.stem
        try:
            self._hash = compute_sha256(self._model_path)
        except Exception as e:
            raise ModelError(f"Failed to compute embedder hash: {e}")

        # Set up ONNX Runtime session
        providers = self._get_providers(config.device)

        try:
            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=providers,
            )
        except Exception as e:
            raise ModelError(f"Failed to load ONNX model: {e}")

        # Get input/output info
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = config.input_size  # (width, height)
        self._embedding_dim = config.embedding_dim

        # ImageNet normalization parameters
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._initialized = False

    def _get_providers(self, device: str) -> List[str]:
        """
        Get ONNX Runtime execution providers.

        Args:
            device: Target device (cuda/cpu/mps)

        Returns:
            List of provider names
        """
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def hash(self) -> str:
        """Model SHA256 hash."""
        return self._hash

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension."""
        return self._embedding_dim

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for model input.

        Args:
            crop: Input crop (H, W, 3), uint8

        Returns:
            Preprocessed tensor (1, 3, H, W), float32
        """
        # Resize to input size
        resized = cv2.resize(crop, self._input_size)

        # Convert to float and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - self._mean) / self._std

        # HWC → CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        return transposed

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for single crop.

        Args:
            crop: RGB/BGR crop, uint8 (H, W, 3)

        Returns:
            L2-normalized embedding (embedding_dim,)
        """
        # Preprocess
        preprocessed = self._preprocess(crop)
        batch = preprocessed[np.newaxis, ...]  # Add batch dimension

        # Run inference
        try:
            outputs = self._session.run(None, {self._input_name: batch})
        except Exception as e:
            raise ModelError(f"Embedder inference failed: {e}")

        embedding = outputs[0][0]  # Remove batch dimension

        # L2 normalize
        norm = np.linalg.norm(embedding) + 1e-8
        embedding = embedding / norm

        return embedding.astype(np.float32)

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple crops.

        Args:
            crops: List of crop images

        Returns:
            List of L2-normalized embeddings
        """
        if not crops:
            return []

        # Preprocess all crops
        preprocessed = [self._preprocess(c) for c in crops]
        batch = np.stack(preprocessed)  # (N, 3, H, W)

        # Run inference
        try:
            outputs = self._session.run(None, {self._input_name: batch})
        except Exception as e:
            raise ModelError(f"Embedder batch inference failed: {e}")

        embeddings = outputs[0]  # (N, embedding_dim)

        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        return [embeddings[i].astype(np.float32) for i in range(len(crops))]

    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        dummy = np.zeros((*self._input_size[::-1], 3), dtype=np.uint8)
        _ = self.embed(dummy)
        self._initialized = True

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "name": self._name,
            "hash": self._hash,
            "hash_short": self._hash[:8],
            "path": str(self._model_path),
            "input_size": self._input_size,
            "embedding_dim": self._embedding_dim,
            "initialized": self._initialized,
        }


class StubEmbedder(BaseEmbedder):
    """
    Stub embedder for testing.

    Returns random or fixed embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        fixed_embedding: Optional[np.ndarray] = None,
        name: str = "stub_embedder",
    ):
        """
        Initialize stub embedder.

        Args:
            embedding_dim: Dimension of embeddings
            fixed_embedding: Fixed embedding to return (random if None)
            name: Model name
        """
        self._embedding_dim = embedding_dim
        self._fixed_embedding = fixed_embedding
        self._name = name
        self._hash = "0" * 64

    @property
    def name(self) -> str:
        return self._name

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return fixed or random embedding."""
        if self._fixed_embedding is not None:
            return self._fixed_embedding.copy()

        # Generate random normalized embedding
        emb = np.random.randn(self._embedding_dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Return embeddings for batch."""
        return [self.embed(c) for c in crops]

    def warmup(self) -> None:
        """No-op for stub."""
        pass

    def set_fixed_embedding(self, embedding: np.ndarray) -> None:
        """Set a fixed embedding to return."""
        self._fixed_embedding = embedding.copy()
