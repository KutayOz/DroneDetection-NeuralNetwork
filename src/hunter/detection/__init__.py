"""
Detection module with hybrid routing architecture.

Components:
- Primary detector (YOLO11)
- Secondary verifier (Siamese)
- Hybrid router for confidence-based routing

Architecture:
    Frame → YOLO11 ──┬── High Conf (>0.8) ──→ Direct to Tracking
                     │
                     ├── Medium Conf (0.5-0.8) → Siamese Verify → Tracking
                     │
                     └── Low Conf (<0.5) ──→ Discard

Stubs (for testing):
- StubDetector: Configurable fake detector
- StubEmbedder: Configurable fake embedder
- StubVerifier: Configurable fake verifier
"""

from .stub import StubDetector, StubEmbedder, StubVerifier
from .yolo11_detector import YOLO11Detector
from .siamese_embedder import SiameseEmbedder
from .siamese_verifier import SiameseVerifier
from .hybrid_detector import HybridDetector

__all__ = [
    # Real implementations
    "YOLO11Detector",
    "SiameseEmbedder",
    "SiameseVerifier",
    "HybridDetector",
    # Stubs for testing
    "StubDetector",
    "StubEmbedder",
    "StubVerifier",
]
