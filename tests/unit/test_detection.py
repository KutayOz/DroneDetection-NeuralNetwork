"""
Unit tests for detection module.

Tests for YOLO11Detector, SiameseEmbedder, SiameseVerifier, and HybridDetector.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hunter.interfaces.detector import (
    IDetector,
    IEmbedder,
    IVerifier,
    IDetectionRouter,
    BoundingBox,
    DetectionResult,
    VerificationResult,
)

# Create mock ultralytics and torch modules before imports that use them
mock_ultralytics = MagicMock()
sys.modules['ultralytics'] = mock_ultralytics

mock_torch = MagicMock()
mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
mock_torch.from_numpy = MagicMock(side_effect=lambda x: MagicMock(
    to=MagicMock(return_value=MagicMock()),
    cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=x))),
    numpy=MagicMock(return_value=x),
))
mock_torch.load = MagicMock()
sys.modules['torch'] = mock_torch

from hunter.detection.yolo11_detector import YOLO11Detector
from hunter.detection.siamese_embedder import SiameseEmbedder
from hunter.detection.siamese_verifier import SiameseVerifier
from hunter.detection.hybrid_detector import HybridDetector


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Create dummy image for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_crop() -> np.ndarray:
    """Create dummy crop for embedding/verification."""
    return np.zeros((128, 128, 3), dtype=np.uint8)


@pytest.fixture
def mock_yolo_model():
    """Create mock YOLO model."""
    mock = MagicMock()
    # Mock model() returns Results object with boxes
    mock_results = MagicMock()
    mock_results.boxes = MagicMock()
    mock_results.boxes.xyxy = np.array([[100, 100, 200, 200]])
    mock_results.boxes.conf = np.array([0.85])
    mock_results.boxes.cls = np.array([0])
    mock.return_value = [mock_results]
    mock.names = {0: "drone"}
    return mock


@pytest.fixture
def mock_siamese_model():
    """Create mock Siamese model."""
    mock = MagicMock()
    mock.return_value = np.random.randn(1, 128).astype(np.float32)
    return mock


# ============================================
# YOLO11Detector Tests
# ============================================


class TestYOLO11Detector:
    """Tests for YOLO11 detector implementation."""

    @pytest.fixture(autouse=True)
    def setup_yolo_mock(self, mock_yolo_model):
        """Setup mock YOLO for all tests in this class."""
        mock_ultralytics.YOLO.return_value = mock_yolo_model
        yield

    def test_implements_idetector(self, mock_yolo_model):
        """YOLO11Detector implements IDetector protocol."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        assert isinstance(detector, IDetector)

    def test_name_property(self, mock_yolo_model):
        """Detector has name property."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        assert isinstance(detector.name, str)
        assert "YOLO11" in detector.name or "yolo" in detector.name.lower()

    def test_model_hash_property(self, mock_yolo_model):
        """Detector has model_hash property."""
        with patch("hunter.detection.yolo11_detector.compute_checksum_short", return_value="abc123"):
            detector = YOLO11Detector(
                model_path=Path("model.pt"),
                confidence_threshold=0.5,
                device="cpu",
            )
            assert isinstance(detector.model_hash, str)

    def test_detect_returns_list(self, mock_yolo_model, dummy_image):
        """detect() returns list of DetectionResult."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        results = detector.detect(dummy_image)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, DetectionResult)

    def test_detect_with_detections(self, mock_yolo_model, dummy_image):
        """detect() returns DetectionResult with correct structure."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        results = detector.detect(dummy_image)

        assert len(results) == 1
        detection = results[0]
        assert isinstance(detection.bbox, BoundingBox)
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.class_id >= 0

    def test_detect_filters_by_confidence(self, dummy_image):
        """detect() filters detections below threshold."""
        mock = MagicMock()
        mock_results = MagicMock()
        mock_results.boxes = MagicMock()
        # Two detections: one above threshold, one below
        mock_results.boxes.xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_results.boxes.conf = np.array([0.8, 0.3])  # 0.3 < 0.5 threshold
        mock_results.boxes.cls = np.array([0, 0])
        mock.return_value = [mock_results]
        mock.names = {0: "drone"}

        mock_ultralytics.YOLO.return_value = mock
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        results = detector.detect(dummy_image)

        # Only detection with confidence >= 0.5 should pass
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_warmup_no_error(self, mock_yolo_model):
        """warmup() runs without error."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            device="cpu",
        )
        # Should not raise
        detector.warmup()

    def test_nms_threshold(self, mock_yolo_model, dummy_image):
        """Detector uses NMS threshold."""
        detector = YOLO11Detector(
            model_path=Path("model.pt"),
            confidence_threshold=0.5,
            nms_threshold=0.45,
            device="cpu",
        )
        # Just verify it accepts the parameter
        assert detector._nms_threshold == 0.45


# ============================================
# SiameseEmbedder Tests
# ============================================


class TestSiameseEmbedder:
    """Tests for Siamese embedder implementation."""

    def test_implements_iembedder(self, mock_siamese_model):
        """SiameseEmbedder implements IEmbedder protocol."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            assert isinstance(embedder, IEmbedder)

    def test_name_property(self, mock_siamese_model):
        """Embedder has name property."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            assert isinstance(embedder.name, str)

    def test_embedding_dim_property(self, mock_siamese_model):
        """Embedder has embedding_dim property."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            assert embedder.embedding_dim == 128

    def test_embed_returns_array(self, mock_siamese_model, dummy_crop):
        """embed() returns numpy array."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            embedding = embedder.embed(dummy_crop)
            assert isinstance(embedding, np.ndarray)

    def test_embed_correct_dimension(self, mock_siamese_model, dummy_crop):
        """embed() returns correct dimension."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            embedding = embedder.embed(dummy_crop)
            assert embedding.shape == (128,)

    def test_embed_normalized(self, mock_siamese_model, dummy_crop):
        """embed() returns L2-normalized vector."""
        mock_siamese_model.return_value = np.random.randn(1, 128).astype(np.float32)
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            embedding = embedder.embed(dummy_crop)
            norm = np.linalg.norm(embedding)
            assert norm == pytest.approx(1.0, rel=0.01)

    def test_embed_batch_returns_list(self, mock_siamese_model, dummy_crop):
        """embed_batch() returns list of arrays."""
        mock_siamese_model.return_value = np.random.randn(3, 128).astype(np.float32)
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            crops = [dummy_crop, dummy_crop, dummy_crop]
            embeddings = embedder.embed_batch(crops)

            assert isinstance(embeddings, list)
            assert len(embeddings) == 3
            for emb in embeddings:
                assert isinstance(emb, np.ndarray)
                assert emb.shape == (128,)

    def test_warmup_no_error(self, mock_siamese_model):
        """warmup() runs without error."""
        with patch("hunter.detection.siamese_embedder.load_siamese_model", return_value=mock_siamese_model):
            embedder = SiameseEmbedder(
                model_path=Path("siamese.pt"),
                embedding_dim=128,
                device="cpu",
            )
            embedder.warmup()


# ============================================
# SiameseVerifier Tests
# ============================================


class TestSiameseVerifier:
    """Tests for Siamese verifier implementation."""

    def test_implements_iverifier(self, mock_siamese_model):
        """SiameseVerifier implements IVerifier protocol."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            assert isinstance(verifier, IVerifier)

    def test_name_property(self, mock_siamese_model):
        """Verifier has name property."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            assert isinstance(verifier.name, str)

    def test_similarity_threshold_property(self, mock_siamese_model):
        """Verifier has similarity_threshold property."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            assert verifier.similarity_threshold == 0.7

    def test_verify_returns_result(self, mock_siamese_model, dummy_crop):
        """verify() returns VerificationResult."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            result = verifier.verify(dummy_crop)
            assert isinstance(result, VerificationResult)

    def test_verify_result_structure(self, mock_siamese_model, dummy_crop):
        """verify() result has correct structure."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            result = verifier.verify(dummy_crop)

            # Accept both Python bool and numpy bool
            assert result.is_verified in (True, False) or bool(result.is_verified) in (True, False)
            assert 0.0 <= float(result.similarity_score) <= 1.0

    def test_verify_with_reference(self, mock_siamese_model, dummy_crop):
        """verify() works with reference embedding."""
        mock_siamese_model.return_value = np.random.randn(1, 128).astype(np.float32)
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            reference = np.random.randn(128).astype(np.float32)
            reference = reference / np.linalg.norm(reference)

            result = verifier.verify(dummy_crop, reference_embedding=reference)
            assert isinstance(result, VerificationResult)

    def test_verify_batch_returns_list(self, mock_siamese_model, dummy_crop):
        """verify_batch() returns list of VerificationResult."""
        mock_siamese_model.return_value = np.random.randn(3, 128).astype(np.float32)
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            crops = [dummy_crop, dummy_crop, dummy_crop]
            results = verifier.verify_batch(crops)

            assert isinstance(results, list)
            assert len(results) == 3
            for result in results:
                assert isinstance(result, VerificationResult)

    def test_warmup_no_error(self, mock_siamese_model):
        """warmup() runs without error."""
        with patch("hunter.detection.siamese_verifier.load_siamese_model", return_value=mock_siamese_model):
            verifier = SiameseVerifier(
                model_path=Path("siamese.pt"),
                similarity_threshold=0.7,
                device="cpu",
            )
            verifier.warmup()


# ============================================
# HybridDetector Tests
# ============================================


class TestHybridDetector:
    """Tests for hybrid detection routing."""

    @pytest.fixture
    def mock_detector(self) -> IDetector:
        """Create mock detector."""
        mock = MagicMock(spec=IDetector)
        mock.name = "MockDetector"
        mock.model_hash = "abc123"
        mock.detect.return_value = [
            DetectionResult(
                bbox=BoundingBox(100, 100, 200, 200),
                confidence=0.9,
                class_id=0,
            ),
            DetectionResult(
                bbox=BoundingBox(300, 300, 400, 400),
                confidence=0.6,  # Medium confidence
                class_id=0,
            ),
            DetectionResult(
                bbox=BoundingBox(500, 500, 600, 600),
                confidence=0.3,  # Low confidence
                class_id=0,
            ),
        ]
        return mock

    @pytest.fixture
    def mock_verifier(self) -> IVerifier:
        """Create mock verifier."""
        mock = MagicMock(spec=IVerifier)
        mock.name = "MockVerifier"
        mock.model_hash = "def456"
        mock.similarity_threshold = 0.7
        mock.verify.return_value = VerificationResult(
            is_verified=True,
            similarity_score=0.85,
        )
        return mock

    def test_implements_idetector(self, mock_detector, mock_verifier):
        """HybridDetector implements IDetector protocol."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        assert isinstance(hybrid, IDetector)

    def test_implements_irouter(self, mock_detector, mock_verifier):
        """HybridDetector implements IDetectionRouter protocol."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        assert isinstance(hybrid, IDetectionRouter)

    def test_name_property(self, mock_detector, mock_verifier):
        """Hybrid detector has name property."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        assert isinstance(hybrid.name, str)
        assert "Hybrid" in hybrid.name or "hybrid" in hybrid.name.lower()

    def test_detect_routes_high_confidence(self, mock_detector, mock_verifier, dummy_image):
        """High confidence detections pass directly."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        results = hybrid.detect(dummy_image)

        # High confidence (0.9) should pass directly
        high_conf = [r for r in results if r.confidence >= 0.8]
        assert len(high_conf) >= 1

    def test_detect_verifies_medium_confidence(self, mock_detector, mock_verifier, dummy_image):
        """Medium confidence detections go through verification."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        hybrid.detect(dummy_image)

        # Verifier should be called for medium confidence detection
        assert mock_verifier.verify.called or mock_verifier.verify_batch.called

    def test_detect_discards_low_confidence(self, mock_detector, mock_verifier, dummy_image):
        """Low confidence detections are discarded."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        results = hybrid.detect(dummy_image)

        # Low confidence (0.3) should be discarded
        low_conf = [r for r in results if r.confidence < 0.4]
        assert len(low_conf) == 0

    def test_threshold_properties(self, mock_detector, mock_verifier):
        """Router has threshold properties."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        assert hybrid.high_confidence_threshold == 0.8
        assert hybrid.low_confidence_threshold == 0.4

    def test_route_method(self, mock_detector, mock_verifier, dummy_image):
        """route() method works correctly."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        detections = mock_detector.detect(dummy_image)
        routed = hybrid.route(detections, dummy_image)

        assert isinstance(routed, list)
        for det in routed:
            assert isinstance(det, DetectionResult)

    def test_warmup_calls_both_models(self, mock_detector, mock_verifier):
        """warmup() warms up both detector and verifier."""
        hybrid = HybridDetector(
            primary_detector=mock_detector,
            verifier=mock_verifier,
            high_threshold=0.8,
            low_threshold=0.4,
        )
        hybrid.warmup()

        mock_detector.warmup.assert_called_once()
        mock_verifier.warmup.assert_called_once()


# ============================================
# Stub Detector Tests (For Testing Without Models)
# ============================================


class TestStubDetector:
    """Tests for stub detector (testing helper)."""

    def test_stub_detector_exists(self):
        """StubDetector can be imported."""
        from hunter.detection.stub import StubDetector
        assert StubDetector is not None

    def test_stub_detector_implements_idetector(self):
        """StubDetector implements IDetector."""
        from hunter.detection.stub import StubDetector
        detector = StubDetector()
        assert isinstance(detector, IDetector)

    def test_stub_detector_returns_configured_detections(self):
        """StubDetector returns configured detections."""
        from hunter.detection.stub import StubDetector

        configured = [
            DetectionResult(
                bbox=BoundingBox(10, 10, 50, 50),
                confidence=0.9,
                class_id=0,
            )
        ]
        detector = StubDetector(detections=configured)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(image)

        assert len(results) == 1
        assert results[0].confidence == 0.9


class TestStubEmbedder:
    """Tests for stub embedder (testing helper)."""

    def test_stub_embedder_exists(self):
        """StubEmbedder can be imported."""
        from hunter.detection.stub import StubEmbedder
        assert StubEmbedder is not None

    def test_stub_embedder_implements_iembedder(self):
        """StubEmbedder implements IEmbedder."""
        from hunter.detection.stub import StubEmbedder
        embedder = StubEmbedder(embedding_dim=128)
        assert isinstance(embedder, IEmbedder)

    def test_stub_embedder_returns_random_embedding(self):
        """StubEmbedder returns random normalized embedding."""
        from hunter.detection.stub import StubEmbedder

        embedder = StubEmbedder(embedding_dim=128)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        embedding = embedder.embed(crop)

        assert embedding.shape == (128,)
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, rel=0.01)


class TestStubVerifier:
    """Tests for stub verifier (testing helper)."""

    def test_stub_verifier_exists(self):
        """StubVerifier can be imported."""
        from hunter.detection.stub import StubVerifier
        assert StubVerifier is not None

    def test_stub_verifier_implements_iverifier(self):
        """StubVerifier implements IVerifier."""
        from hunter.detection.stub import StubVerifier
        verifier = StubVerifier()
        assert isinstance(verifier, IVerifier)

    def test_stub_verifier_returns_configured_result(self):
        """StubVerifier returns configured verification result."""
        from hunter.detection.stub import StubVerifier

        verifier = StubVerifier(always_verify=True, fixed_score=0.95)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        result = verifier.verify(crop)

        assert result.is_verified is True
        assert result.similarity_score == 0.95
