# File: src/walta/walta_llm/providers/base.py
# This file defines the base LLM provider class and configuration dataclasses.

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    Matches parameters in LLMProviderProtocol's get_completion.
    """
    temperature: float = 0.7
    max_tokens: int = 1000
    # Add other parameters if needed, but stick to protocol for now

@dataclass
class EmbedConfig:
    """
    Configuration for text embeddings.
    """
    model: str = "embedding-001"

class LLMProvider:
    """
    Abstract base class for LLM providers.
    Concrete providers (like OpenAIProvider, GeminiProvider) should inherit from this
    and implement the methods defined in LLMProviderProtocol.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError("This method must be implemented by subclasses.")
    
    async def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")
    
    async def get_multimodal_analysis(
        self,
        text: str,
        image: Optional[bytes] = None
    ) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")
