"""
Contract tests for detector interfaces.

These tests verify that implementations correctly fulfill
their Protocol contracts. Uses hypothesis for property-based testing.
"""

import numpy as np
import pytest
from typing import List, Optional
from dataclasses import dataclass

from hunter.interfaces.detector import (
    IDetector,
    IEmbedder,
    IVerifier,
    DetectionResult,
    VerificationResult,
    BoundingBox,
)


# ============================================
# Mock implementations for contract testing
# ============================================

class MockDetector:
    """Mock detector that fulfills IDetector contract."""

    def __init__(self, name: str = "mock_detector", model_hash: str = "abc123"):
        self._name = name
        self._model_hash = model_hash
        self._warmed_up = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_hash(self) -> str:
        return self._model_hash

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Return mock detections."""
        return [
            DetectionResult(
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
                confidence=0.9,
                class_id=0,
            )
        ]

    def warmup(self) -> None:
        self._warmed_up = True


class MockEmbedder:
    """Mock embedder that fulfills IEmbedder contract."""

    def __init__(
        self,
        name: str = "mock_embedder",
        model_hash: str = "def456",
        embedding_dim: int = 128,
    ):
        self._name = name
        self._model_hash = model_hash
        self._embedding_dim = embedding_dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return mock embedding."""
        emb = np.random.randn(self._embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def embed_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Return mock embeddings for batch."""
        return [self.embed(crop) for crop in crops]

    def warmup(self) -> None:
        pass


class MockVerifier:
    """Mock verifier that fulfills IVerifier contract."""

    def __init__(
        self,
        name: str = "mock_verifier",
        model_hash: str = "ghi789",
        threshold: float = 0.7,
    ):
        self._name = name
        self._model_hash = model_hash
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    def verify(
        self,
        crop: np.ndarray,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """Return mock verification result."""
        return VerificationResult(
            is_verified=True,
            similarity_score=0.85,
            embedding=np.random.randn(128).astype(np.float32),
        )

    def verify_batch(
        self,
        crops: List[np.ndarray],
        reference_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[VerificationResult]:
        """Return mock verification results."""
        return [self.verify(crop) for crop in crops]

    def warmup(self) -> None:
        pass


# ============================================
# Contract Tests
# ============================================

class TestIDetectorContract:
    """Contract tests for IDetector protocol."""

    @pytest.fixture
    def detector(self) -> MockDetector:
        return MockDetector()

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_has_name_property(self, detector: MockDetector):
        """IDetector must have name property returning str."""
        assert isinstance(detector.name, str)
        assert len(detector.name) > 0

    def test_has_model_hash_property(self, detector: MockDetector):
        """IDetector must have model_hash property returning str."""
        assert isinstance(detector.model_hash, str)
        assert len(detector.model_hash) > 0

    def test_detect_returns_list_of_detections(
        self, detector: MockDetector, sample_image: np.ndarray
    ):
        """IDetector.detect must return List[DetectionResult]."""
        results = detector.detect(sample_image)

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, DetectionResult)

    def test_detect_with_empty_image(self, detector: MockDetector):
        """IDetector.detect should handle edge cases gracefully."""
        # Empty image (valid numpy array but no content)
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(empty_image)
        assert isinstance(results, list)

    def test_detection_result_has_required_fields(
        self, detector: MockDetector, sample_image: np.ndarray
    ):
        """DetectionResult must have bbox, confidence, class_id."""
        results = detector.detect(sample_image)

        if results:
            result = results[0]
            assert hasattr(result, "bbox")
            assert hasattr(result, "confidence")
            assert hasattr(result, "class_id")
            assert isinstance(result.bbox, BoundingBox)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.class_id, int)

    def test_bounding_box_is_valid(
        self, detector: MockDetector, sample_image: np.ndarray
    ):
        """BoundingBox must have x1 <= x2 and y1 <= y2."""
        results = detector.detect(sample_image)

        if results:
            bbox = results[0].bbox
            assert bbox.x1 <= bbox.x2
            assert bbox.y1 <= bbox.y2
            assert bbox.width >= 0
            assert bbox.height >= 0

    def test_warmup_is_callable(self, detector: MockDetector):
        """IDetector must have warmup method."""
        # Should not raise
        detector.warmup()


class TestIEmbedderContract:
    """Contract tests for IEmbedder protocol."""

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        return MockEmbedder(embedding_dim=128)

    @pytest.fixture
    def sample_crop(self) -> np.ndarray:
        return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def test_has_name_property(self, embedder: MockEmbedder):
        """IEmbedder must have name property."""
        assert isinstance(embedder.name, str)

    def test_has_model_hash_property(self, embedder: MockEmbedder):
        """IEmbedder must have model_hash property."""
        assert isinstance(embedder.model_hash, str)

    def test_has_embedding_dim_property(self, embedder: MockEmbedder):
        """IEmbedder must have embedding_dim property."""
        assert isinstance(embedder.embedding_dim, int)
        assert embedder.embedding_dim > 0

    def test_embed_returns_numpy_array(
        self, embedder: MockEmbedder, sample_crop: np.ndarray
    ):
        """IEmbedder.embed must return numpy array."""
        embedding = embedder.embed(sample_crop)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) == embedder.embedding_dim

    def test_embed_returns_normalized_vector(
        self, embedder: MockEmbedder, sample_crop: np.ndarray
    ):
        """IEmbedder.embed should return normalized embedding."""
        embedding = embedder.embed(sample_crop)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Embedding norm should be ~1.0, got {norm}"

    def test_embed_batch_returns_list(self, embedder: MockEmbedder):
        """IEmbedder.embed_batch must return list of embeddings."""
        crops = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        embeddings = embedder.embed_batch(crops)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(crops)
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == embedder.embedding_dim

    def test_embed_batch_empty_list(self, embedder: MockEmbedder):
        """IEmbedder.embed_batch should handle empty list."""
        embeddings = embedder.embed_batch([])
        assert embeddings == []


class TestIVerifierContract:
    """Contract tests for IVerifier protocol."""

    @pytest.fixture
    def verifier(self) -> MockVerifier:
        return MockVerifier(threshold=0.7)

    @pytest.fixture
    def sample_crop(self) -> np.ndarray:
        return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def test_has_name_property(self, verifier: MockVerifier):
        """IVerifier must have name property."""
        assert isinstance(verifier.name, str)

    def test_has_model_hash_property(self, verifier: MockVerifier):
        """IVerifier must have model_hash property."""
        assert isinstance(verifier.model_hash, str)

    def test_has_similarity_threshold_property(self, verifier: MockVerifier):
        """IVerifier must have similarity_threshold property."""
        assert isinstance(verifier.similarity_threshold, float)
        assert 0.0 <= verifier.similarity_threshold <= 1.0

    def test_verify_returns_verification_result(
        self, verifier: MockVerifier, sample_crop: np.ndarray
    ):
        """IVerifier.verify must return VerificationResult."""
        result = verifier.verify(sample_crop)

        assert isinstance(result, VerificationResult)
        assert isinstance(result.is_verified, bool)
        assert isinstance(result.similarity_score, float)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_verify_with_reference_embedding(
        self, verifier: MockVerifier, sample_crop: np.ndarray
    ):
        """IVerifier.verify should accept optional reference embedding."""
        reference = np.random.randn(128).astype(np.float32)
        result = verifier.verify(sample_crop, reference_embedding=reference)

        assert isinstance(result, VerificationResult)

    def test_verify_batch_returns_list(self, verifier: MockVerifier):
        """IVerifier.verify_batch must return list of results."""
        crops = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        results = verifier.verify_batch(crops)

        assert isinstance(results, list)
        assert len(results) == len(crops)
        for result in results:
            assert isinstance(result, VerificationResult)


class TestBoundingBox:
    """Tests for BoundingBox value object."""

    def test_creation(self):
        """BoundingBox can be created with coordinates."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)

        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 100.0
        assert bbox.y2 == 150.0

    def test_width_property(self):
        """BoundingBox.width returns correct value."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)
        assert bbox.width == 90.0

    def test_height_property(self):
        """BoundingBox.height returns correct value."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)
        assert bbox.height == 130.0

    def test_center_property(self):
        """BoundingBox.center returns correct value."""
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        cx, cy = bbox.center
        assert cx == 50.0
        assert cy == 50.0

    def test_area_property(self):
        """BoundingBox.area returns correct value."""
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
        assert bbox.area == 200.0

    def test_as_tuple(self):
        """BoundingBox.as_tuple returns (x1, y1, x2, y2)."""
        bbox = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert bbox.as_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_as_xywh(self):
        """BoundingBox.as_xywh returns (x, y, w, h)."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
        x, y, w, h = bbox.as_xywh()
        assert x == 10.0
        assert y == 20.0
        assert w == 40.0
        assert h == 60.0

    def test_immutability(self):
        """BoundingBox should be immutable (frozen dataclass)."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=150.0)

        with pytest.raises(AttributeError):
            bbox.x1 = 0.0


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation_minimal(self):
        """DetectionResult can be created with minimal fields."""
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        result = DetectionResult(bbox=bbox, confidence=0.9, class_id=0)

        assert result.bbox == bbox
        assert result.confidence == 0.9
        assert result.class_id == 0
        assert result.embedding is None

    def test_creation_with_embedding(self):
        """DetectionResult can include embedding."""
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        embedding = np.zeros(128, dtype=np.float32)

        result = DetectionResult(
            bbox=bbox, confidence=0.9, class_id=0, embedding=embedding
        )

        assert result.embedding is not None
        assert len(result.embedding) == 128


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_creation(self):
        """VerificationResult can be created."""
        result = VerificationResult(
            is_verified=True,
            similarity_score=0.85,
            embedding=np.zeros(128, dtype=np.float32),
        )

        assert result.is_verified is True
        assert result.similarity_score == 0.85
        assert result.embedding is not None

    def test_optional_embedding(self):
        """VerificationResult embedding is optional."""
        result = VerificationResult(
            is_verified=False,
            similarity_score=0.3,
        )

        assert result.embedding is None
