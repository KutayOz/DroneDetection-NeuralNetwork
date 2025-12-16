"""
Appearance embedding models.

Provides Siamese network embedder for track association.
"""

from .base import BaseEmbedder
from .siamese_embedder import SiameseEmbedder, StubEmbedder

__all__ = [
    "BaseEmbedder",
    "SiameseEmbedder",
    "StubEmbedder",
]
