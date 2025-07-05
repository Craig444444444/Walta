# src/walta/walta_llm/providers/base.py
# This file now defines LLMProviderProtocol (including get_embedding), LLMProvider, and error classes.

import abc
from typing import Protocol, Dict, Any, List, TypedDict, Optional, Union
from dataclasses import dataclass # <--- ADD THIS LINE

class LLMGenerationError(Exception):
    """
    Custom exception for errors during LLM text generation.
    This can wrap an underlying API error for more unified handling.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class ChatMessage(TypedDict):
    role: str
    content: str

class LLMProviderProtocol(Protocol):
    """
    A protocol defining the interface for all Large Language Model (LLM) providers.
    Any class implementing this protocol must provide the specified methods
    and adhere to their signatures.
    """

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text based on a given prompt.
        """
        ...

    @abc.abstractmethod
    def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """
        Generates a chat completion based on a list of chat messages.
        """
        ...

    @abc.abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters for the LLM model.
        """
        ...

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string.
        """
        ...

    @abc.abstractmethod
    def get_multimodal_analysis(
        self,
        text: str,
        image: Optional[bytes] = None
    ) -> str:
        """
        Performs multimodal analysis (text and optional image).
        """
        ...

    @abc.abstractmethod
    def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generates embedding vectors for a given text or list of texts.
        The return type is List[List[float]] to support batch embeddings.
        """
        ...


# --- Configuration Dataclasses (keep these) ---
@dataclass # Now dataclass is imported!
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 1000

@dataclass
class EmbedConfig:
    """Configuration for text embeddings."""
    model: str = "embedding-001"


# --- Base LLM Provider Class ---
class LLMProvider(LLMProviderProtocol): # Explicitly state it implements the protocol
    """
    Abstract base class for LLM providers.
    Concrete providers (like OpenAIProvider, GeminiProvider) should inherit from this
    and implement the methods defined in LLMProviderProtocol.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    # Implement all abstract methods from LLMProviderProtocol with NotImplementedError
    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def get_model_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def count_tokens(self, text: str) -> int:
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def get_multimodal_analysis(self, text: str, image: Optional[bytes] = None) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")

    async def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        raise NotImplementedError("This method must be implemented by subclasses.")
