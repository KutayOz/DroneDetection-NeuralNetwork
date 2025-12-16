"""
Siamese network verifier implementation.

Secondary verification for uncertain detections using Siamese comparison.

Follows Single Responsibility Principle (SRP):
- Only handles verification decisions
- Uses embedder for feature extraction
- Binary classification based on similarity
"""

from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np

from ..interfaces.detector import IVerifier, VerificationResult
from ..utils.checksum import compute_checksum_short


def load_siamese_model(model_path: Path, device: str = "cuda"):
    """
    Load Siamese model from file.

    Args:
        model_path: Path to model file
        device: Inference device

    Returns:
        Loaded model
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


class SiameseVerifier:
    """
    Siamese network verifier for secondary detection verification.

    Compares detection crops against reference embeddings or learned
    drone prototypes to verify if detection is a true drone.
    """

    def __init__(
        self,
        model_path: Path,
        similarity_threshold: float = 0.7,
        input_size: tuple = (128, 128),
        embedding_dim: int = 128,
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        reference_embeddings: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Initialize Siamese verifier.

        Args:
            model_path: Path to Siamese model file
            similarity_threshold: Threshold for positive verification
            input_size: Model input size (width, height)
            embedding_dim: Embedding dimension
            device: Inference device
            reference_embeddings: Pre-computed reference drone embeddings
        """
        self._model_path = Path(model_path)
        self._similarity_threshold = similarity_threshold
        self._input_size = input_size
        self._embedding_dim = embedding_dim
        self._device = device
        self._reference_embeddings = reference_embeddings or []

        # Normalization parameters
        self._normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Lazy load
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
        return f"SiameseVerifier-{self._model_path.stem}"

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
    def similarity_threshold(self) -> float:
        """Threshold for positive verification."""
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        """Update similarity threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._similarity_threshold = value

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

        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self._normalize_mean) / self._normalize_std

        # HWC to CHW, add batch dimension
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)

        return batch.astype(np.float32)

    def _compute_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute embedding for crop.

        Args:
            crop: Input crop

        Returns:
            Normalized embedding vector
        """
        input_tensor = self._preprocess(crop)

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

        embedding = output.flatten()[:self._embedding_dim]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            emb1: First embedding (should be normalized)
            emb2: Second embedding (should be normalized)

        Returns:
            Similarity score in [0, 1]
        """
        # For normalized vectors, dot product = cosine similarity
        sim = np.dot(emb1, emb2)
        # Map from [-1, 1] to [0, 1]
        return (sim + 1.0) / 2.0

    def verify(
        self,
        crop: np.ndarray,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Verify if crop contains target object.

        Args:
            crop: Cropped image to verify
            reference_embedding: Optional reference for comparison

        Returns:
            VerificationResult with decision and score
        """
        self._ensure_model_loaded()

        # Compute embedding for crop
        crop_embedding = self._compute_embedding(crop)

        # Determine references to compare against
        references = []
        if reference_embedding is not None:
            references.append(reference_embedding)
        references.extend(self._reference_embeddings)

        # If no references, use learned prototype or return low confidence
        if len(references) == 0:
            # Use embedding norm as a weak proxy for "droneness"
            # (well-trained model should produce consistent embeddings for drones)
            norm_score = min(1.0, np.linalg.norm(crop_embedding))
            return VerificationResult(
                is_verified=norm_score >= self._similarity_threshold,
                similarity_score=norm_score,
                embedding=crop_embedding,
            )

        # Compute max similarity across all references
        max_similarity = 0.0
        for ref in references:
            sim = self._cosine_similarity(crop_embedding, ref)
            max_similarity = max(max_similarity, sim)

        is_verified = max_similarity >= self._similarity_threshold

        return VerificationResult(
            is_verified=is_verified,
            similarity_score=max_similarity,
            embedding=crop_embedding,
        )

    def verify_batch(
        self,
        crops: List[np.ndarray],
        reference_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[VerificationResult]:
        """
        Verify multiple crops (batched for efficiency).

        Args:
            crops: List of cropped images
            reference_embeddings: Optional list of references

        Returns:
            List of VerificationResult objects
        """
        if len(crops) == 0:
            return []

        self._ensure_model_loaded()

        # Compute embeddings for all crops
        inputs = [self._preprocess(crop) for crop in crops]
        batch = np.concatenate(inputs, axis=0)

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

        # Process each result
        results = []
        for i in range(len(crops)):
            embedding = outputs[i].flatten()[:self._embedding_dim]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Get reference for this crop if provided
            ref = None
            if reference_embeddings and i < len(reference_embeddings):
                ref = reference_embeddings[i]

            # Compute similarity
            references = []
            if ref is not None:
                references.append(ref)
            references.extend(self._reference_embeddings)

            if len(references) == 0:
                norm_score = min(1.0, np.linalg.norm(embedding))
                results.append(VerificationResult(
                    is_verified=norm_score >= self._similarity_threshold,
                    similarity_score=norm_score,
                    embedding=embedding,
                ))
            else:
                max_sim = max(self._cosine_similarity(embedding, r) for r in references)
                results.append(VerificationResult(
                    is_verified=max_sim >= self._similarity_threshold,
                    similarity_score=max_sim,
                    embedding=embedding,
                ))

        return results

    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        self._ensure_model_loaded()
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        self.verify(dummy)

    def add_reference(self, embedding: np.ndarray) -> None:
        """
        Add reference embedding for comparison.

        Args:
            embedding: Pre-computed reference embedding
        """
        self._reference_embeddings.append(embedding)

    def clear_references(self) -> None:
        """Clear all reference embeddings."""
        self._reference_embeddings = []
