# File: src/walta/walta_llm/providers/__init__.py
# THIS FILE MUST BE RENAMED TO __init__.py FROM __init__.
# This content combines your LLMProviderProtocol definition with necessary imports and __all__ exports.

"""
Walta LLM Providers - Contains implementations for different LLM APIs.
"""

from typing import Protocol, List, Dict, Any, Optional, Union

# Import base classes and configs from base.py
from .base import LLMProvider, GenerationConfig, EmbedConfig
# Import concrete provider implementations
from .openai import OpenAIProvider
from .gemini import GeminiProvider


class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers."""
    async def get_embedding(self, text: str) -> List[float]: ...
    
    async def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str: ...
    
    async def get_multimodal_analysis(
        self,
        text: str,
        image: Optional[bytes] = None
    ) -> str: ...

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class ProviderError(LLMError):
    """Exception for provider-specific errors."""
    pass

class EmbeddingError(LLMError):
    """Exception for embedding-related errors."""
    pass

class CompletionError(LLMError):
    """Exception for completion-related errors."""
    pass

# Ensure all necessary components are exposed for external imports
__all__ = [
    "LLMProviderProtocol",
    "LLMError",
    "ProviderError",
    "EmbeddingError",
    "CompletionError",
    "LLMProvider",      # From base.py
    "GenerationConfig", # From base.py
    "EmbedConfig",      # From base.py
    "OpenAIProvider",
    "GeminiProvider"
]
