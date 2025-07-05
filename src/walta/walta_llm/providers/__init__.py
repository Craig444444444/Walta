# src/walta/walta_llm/providers/__init__.py
"""
Walta LLM Providers - Contains implementations for different LLM APIs.
"""

# Import base classes, protocol, and error classes from base.py
# These are the foundational elements, so they come first.
from .base import (
    LLMProvider,
    LLMProviderProtocol,
    LLMGenerationError # Renamed from LLMError, ProviderError, etc. for simplicity
)

# Import concrete provider implementations
# These depend on elements from base.py, so they are imported after base.py is ready.
from .openai import OpenAIProvider
from .gemini import GeminiProvider


# Ensure all necessary components are exposed for external imports
# This list controls what gets imported when someone does `from walta.walta_llm.providers import *`
__all__ = [
    "LLMProvider",
    "LLMProviderProtocol",
    "LLMGenerationError", # Re-export the unified error class
    "OpenAIProvider",
    "GeminiProvider"
]
