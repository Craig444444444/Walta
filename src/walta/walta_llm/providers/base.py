# File: src/walta/walta_llm/providers/base.py
from typing import Protocol, Dict, Any, List, Union
from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    """
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False # Whether to stream responses

@dataclass
class EmbedConfig:
    """
    Configuration for text embeddings.
    """
    model: str = "embedding-001" # A common default for embedding models

class LLMProviderProtocol(Protocol):
    """
    Protocol defining the core interface for an LLM provider.
    Any class implementing this protocol must provide these methods.
    """
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """
        Generates text based on a prompt and configuration.
        """
        ...

    def embed_text(self, text: Union[str, List[str]], config: EmbedConfig) -> Any:
        """
        Generates embeddings for single or multiple text inputs.
        Returns a list of embeddings.
        """
        ...
    
    # You might also add methods for chat, if applicable
    # def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
    #     """
    #     Handles multi-turn conversational chat.
    #     Messages should be in a format like [{"role": "user", "content": "..."}]
    #     """
    #     ...

class LLMProvider:
    """
    Abstract base class for LLM providers.
    Concrete providers (like OpenAIProvider, GeminiProvider) should inherit from this
    and implement the required methods.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        raise NotImplementedError("This method must be implemented by subclasses.")

    def embed_text(self, text: Union[str, List[str]], config: EmbedConfig) -> Any:
        raise NotImplementedError("This method must be implemented by subclasses.")

    # def chat(self, messages: List[Dict[str, str]], config: GenerationConfig) -> str:
    #     raise NotImplementedError("This method must be implemented by subclasses.")
