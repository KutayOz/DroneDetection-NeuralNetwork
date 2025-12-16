"""
Abstract embedder interface.

Embedders generate appearance feature vectors for:
- Track-detection association
- Hard-negative rejection
- Re-identification after occlusion

Follows ISP: Minimal interface for all embedder implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for appearance embedders.

    Embedders extract feature vectors from image crops
    that can be compared using cosine similarity.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @property
    @abstractmethod
    def hash(self) -> str:
        """Model checksum (SHA256)."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of output embedding vector."""
        pass

    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Generate embedding for single crop.

        Args:
            crop: RGB/BGR crop image, uint8 (H, W, 3)

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple crops.

        More efficient than individual calls for batches.

        Args:
            crops: List of crop images

        Returns:
            List of L2-normalized embedding vectors
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """
        Warm up model.

        First inference is slow due to initialization.
        Call before timing-critical inference.
        """
        pass

    def compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between embeddings.

        Assumes embeddings are L2-normalized.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score in [-1, 1] (1 = identical)
        """
        return float(np.dot(emb1, emb2))

    def compute_distance(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """
        Compute cosine distance between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Distance in [0, 2] (0 = identical)
        """
        return 1.0 - self.compute_similarity(emb1, emb2)

    def compute_similarity_matrix(
        self,
        embeddings1: List[np.ndarray],
        embeddings2: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix of shape (len(embeddings1), len(embeddings2))
        """
        if not embeddings1 or not embeddings2:
            return np.zeros((len(embeddings1), len(embeddings2)), dtype=np.float32)

        # Stack embeddings
        emb1 = np.stack(embeddings1)  # (N, D)
        emb2 = np.stack(embeddings2)  # (M, D)

        # Compute dot product (cosine similarity for L2-normalized vectors)
        similarity = emb1 @ emb2.T  # (N, M)

        return similarity.astype(np.float32)

    @property
    def version(self) -> str:
        """Short version identifier."""
        return self.hash[:8] if self.hash else "unknown"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"dim={self.embedding_dim}, version={self.version})"
        )
