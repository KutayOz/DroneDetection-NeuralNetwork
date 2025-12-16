"""
Stub implementations for testing.

Provides configurable fake implementations of IDetector, IEmbedder, IVerifier
for unit testing without requiring real models.

Follows Liskov Substitution Principle (LSP):
- Stubs are fully substitutable for real implementations
- Can be injected anywhere interfaces are expected
"""

from typing import List, Optional

import numpy as np

from ..interfaces.detector import (
    IDetector,
    IEmbedder,
    IVerifier,
    BoundingBox,
    DetectionResult,
    VerificationResult,
)


class StubDetector:
    """
    Configurable stub detector for testing.

    Returns pre-configured detections regardless of input.
    Useful for testing tracking and pipeline components.
    """

    def __init__(
        self,
        detections: Optional[List[DetectionResult]] = None,
        name: str = "StubDetector",
    ) -> None:
        """
        Initialize stub detector.

        Args:
            detections: Pre-configured detections to return
            name: Detector name
        """
        self._name = name
        self._model_hash = "stub_no_model"
        self._detections = detections or []

    @property
    def name(self) -> str:
        """Model identifier name."""
        return self._name

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        return self._model_hash

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Return pre-configured detections.

        Args:
            image: Input image (ignored)

        Returns:
            Pre-configured detection list
        """
        return self._detections.copy()

    def warmup(self) -> None:
        """No-op warmup for stub."""
        pass

    def set_detections(self, detections: List[DetectionResult]) -> None:
        """
        Update configured detections.

        Args:
            detections: New detections to return
        """
        self._detections = detections


class StubEmbedder:
    """
    Configurable stub embedder for testing.

    Returns random or fixed embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        fixed_embedding: Optional[np.ndarray] = None,
        name: str = "StubEmbedder",
    ) -> None:
        """
        Initialize stub embedder.

        Args:
            embedding_dim: Dimension of embedding vectors
            fixed_embedding: If provided, always return this embedding
            name: Embedder name
        """
        self._name = name
        self._model_hash = "stub_no_model"
        self._embedding_dim = embedding_dim
        self._fixed_embedding = fixed_embedding

    @property
    def name(self) -> str:
        """Model identifier name."""
        return self._name

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        return self._model_hash

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embedding vector."""
        return self._embedding_dim

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for crop.

        Args:
            crop: Input crop (ignored)

        Returns:
            Normalized embedding vector
        """
        if self._fixed_embedding is not None:
            return self._fixed_embedding.copy()

        # Generate random normalized embedding
        embedding = np.random.randn(self._embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple crops.

        Args:
            crops: List of crops (ignored)

        Returns:
            List of normalized embedding vectors
        """
        return [self.embed(crop) for crop in crops]

    def warmup(self) -> None:
        """No-op warmup for stub."""
        pass


class StubVerifier:
    """
    Configurable stub verifier for testing.

    Returns fixed verification results.
    """

    def __init__(
        self,
        always_verify: bool = True,
        fixed_score: float = 0.9,
        similarity_threshold: float = 0.7,
        name: str = "StubVerifier",
    ) -> None:
        """
        Initialize stub verifier.

        Args:
            always_verify: If True, always verify positively
            fixed_score: Fixed similarity score to return
            similarity_threshold: Verification threshold
            name: Verifier name
        """
        self._name = name
        self._model_hash = "stub_no_model"
        self._always_verify = always_verify
        self._fixed_score = fixed_score
        self._similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        """Model identifier name."""
        return self._name

    @property
    def model_hash(self) -> str:
        """Model file checksum."""
        return self._model_hash

    @property
    def similarity_threshold(self) -> float:
        """Threshold for positive verification."""
        return self._similarity_threshold

    def verify(
        self,
        crop: np.ndarray,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Verify crop.

        Args:
            crop: Input crop (ignored)
            reference_embedding: Reference embedding (ignored)

        Returns:
            Pre-configured verification result
        """
        return VerificationResult(
            is_verified=self._always_verify,
            similarity_score=self._fixed_score,
        )

    def verify_batch(
        self,
        crops: List[np.ndarray],
        reference_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[VerificationResult]:
        """
        Verify multiple crops.

        Args:
            crops: List of crops (ignored)
            reference_embeddings: Reference embeddings (ignored)

        Returns:
            List of pre-configured verification results
        """
        return [self.verify(crop) for crop in crops]

    def warmup(self) -> None:
        """No-op warmup for stub."""
        pass

    def set_result(self, is_verified: bool, score: float) -> None:
        """
        Update result configuration.

        Args:
            is_verified: New verification decision
            score: New similarity score
        """
        self._always_verify = is_verified
        self._fixed_score = score
