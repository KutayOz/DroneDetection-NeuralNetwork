"""
LLM integration module for training advisor.

Contains providers for LLM-based analysis.
"""

from .stub_provider import StubLLMProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "StubLLMProvider",
    "OpenAIProvider",
]
