"""
Siamese network embedder implementation.

Generates appearance embeddings for re-identification and tracking.

Follows Single Responsibility Principle (SRP):
- Only handles embedding extraction
- Preprocessing handled internally
- No verification logic
"""

from pathlib import Path
from typing import List, Literal, Optional, Callable

import cv2
import numpy as np

from ..interfaces.detector import IEmbedder
from ..utils.checksum import compute_checksum_short


def load_siamese_model(model_path: Path, device: str = "cuda"):
    """
    Load Siamese model from file.

    Supports PyTorch (.pt) and ONNX (.onnx) formats.

    Args:
        model_path: Path to model file
        device: Inference device

    Returns:
        Loaded model (PyTorch Module or ONNX session)
    """
    suffix = model_path.suffix.lower()

    if suffix == ".pt":
        import torch
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
        return model

    elif suffix == ".onnx":
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        return session

    else:
        raise ValueError(f"Unsupported model format: {suffix}")


class SiameseEmbedder:
    """
    Siamese network embedder for appearance features.

    Generates normalized embedding vectors for object crops.
    Used for re-identification in tracking and verification.
    """

    def __init__(
        self,
        model_path: Path,
        embedding_dim: int = 128,
        input_size: tuple = (128, 128),
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Initialize Siamese embedder.

        Args:
            model_path: Path to Siamese model file
            embedding_dim: Output embedding dimension
            input_size: Model input size (width, height)
            device: Inference device
            normalize_mean: ImageNet normalization mean
            normalize_std: ImageNet normalization std
        """
        self._model_path = Path(model_path)
        self._embedding_dim = embedding_dim
        self._input_size = input_size
        self._device = device
        self._normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self._normalize_std = np.array(normalize_std, dtype=np.float32)

        # Lazy load model
        self._model = None
        self._model_hash_cached: Optional[str] = None
        self._is_onnx = self._model_path.suffix.lower() == ".onnx"

    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._model is None:
            self._model = load_siamese_model(self._model_path, self._device)

    @property
    def name(self) -> str:
        """Model identifier name."""
        return f"SiameseEmbedder-{self._model_path.stem}"

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        if self._model_hash_cached is None:
            if self._model_path.exists():
                self._model_hash_cached = compute_checksum_short(
                    self._model_path, length=12
                )
            else:
                self._model_hash_cached = "model_not_found"
        return self._model_hash_cached

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embedding vector."""
        return self._embedding_dim

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for model input.

        Args:
            crop: Input crop (HWC, BGR uint8)

        Returns:
            Preprocessed tensor (1, C, H, W) float32
        """
        # Resize
        resized = cv2.resize(crop, self._input_size)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - self._normalize_mean) / self._normalize_std

        # HWC to CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)

        return batch.astype(np.float32)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize embedding vector.

        Args:
            embedding: Raw embedding

        Returns:
            L2-normalized embedding (norm = 1)
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for a single crop.

        Args:
            crop: Cropped detection image (HWC format, BGR)

        Returns:
            Normalized embedding vector (L2 norm = 1)
        """
        self._ensure_model_loaded()

        # Preprocess
        input_tensor = self._preprocess(crop)

        # Run inference
        if self._is_onnx:
            input_name = self._model.get_inputs()[0].name
            output = self._model.run(None, {input_name: input_tensor})[0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(input_tensor)
                if self._device != "cpu":
                    tensor = tensor.to(self._device)
                output = self._model(tensor)
                if hasattr(output, 'cpu'):
                    output = output.cpu().numpy()

        # Extract and normalize
        embedding = output.flatten()[:self._embedding_dim]
        return self._normalize_embedding(embedding)

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple crops (batched for efficiency).

        Args:
            crops: List of cropped images

        Returns:
            List of normalized embedding vectors
        """
        if len(crops) == 0:
            return []

        self._ensure_model_loaded()

        # Preprocess all crops
        inputs = [self._preprocess(crop) for crop in crops]
        batch = np.concatenate(inputs, axis=0)

        # Run batched inference
        if self._is_onnx:
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: batch})[0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(batch)
                if self._device != "cpu":
                    tensor = tensor.to(self._device)
                outputs = self._model(tensor)
                if hasattr(outputs, 'cpu'):
                    outputs = outputs.cpu().numpy()

        # Extract and normalize each embedding
        embeddings = []
        for i in range(len(crops)):
            embedding = outputs[i].flatten()[:self._embedding_dim]
            embeddings.append(self._normalize_embedding(embedding))

        return embeddings

    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        self._ensure_model_loaded()
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        self.embed(dummy)
