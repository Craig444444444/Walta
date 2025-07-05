# src/walta/walta_llm/providers/base.py
import abc
from typing import Protocol, Dict, Any, List, TypedDict

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

        Args:
            prompt (str): The input prompt for text generation.
            **kwargs: Additional keyword arguments specific to the LLM provider
                      (e.g., temperature, max_tokens, stop_sequences).

        Returns:
            str: The generated text.

        Raises:
            LLMGenerationError: If an error occurs during text generation.
        """
        ... # Ellipsis indicates an abstract method with no implementation

    @abc.abstractmethod
    def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """
        Generates a chat completion based on a list of chat messages.

        Args:
            messages (List[ChatMessage]): A list of chat messages, each with a 'role' and 'content'.
            **kwargs: Additional keyword arguments specific to the LLM provider.

        Returns:
            str: The generated response from the chat model.

        Raises:
            LLMGenerationError: If an error occurs during chat completion.
        """
        ...

    @abc.abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters for the LLM model.
        This could include model name, default temperature, max tokens, etc.

        Returns:
            Dict[str, Any]: A dictionary of model parameters.
        """
        ...

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string.

        Args:
            text (str): The text string to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        ...
