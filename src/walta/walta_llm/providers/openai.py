# src/walta/walta_llm/providers/openai.py
import os
import openai
from typing import Dict, Any, List

# Import from the base module within the same package
from .base import LLMProviderProtocol, LLMGenerationError, ChatMessage

class OpenAIProvider(LLMProviderProtocol):
    """
    A concrete implementation of LLMProviderProtocol for OpenAI's API.
    """
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", **kwargs: Any):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. "
                    "Please pass it as an argument or set the OPENAI_API_KEY environment variable."
                )
        openai.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.default_params = kwargs

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text using OpenAI's completion API.

        Args:
            prompt (str): The input prompt.
            **kwargs: Overrides for default model parameters.

        Returns:
            str: The generated text.

        Raises:
            LLMGenerationError: If the API call fails.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat_completion(messages, **kwargs)

    def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """
        Generates a chat completion using OpenAI's chat completion API.

        Args:
            messages (List[ChatMessage]): A list of chat messages.
            **kwargs: Overrides for default model parameters.

        Returns:
            str: The generated response.

        Raises:
            LLMGenerationError: If the API call fails.
        """
        params = {**self.default_params, **kwargs}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            # Accessing the content from the first choice
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                raise LLMGenerationError("OpenAI API returned an empty or malformed response.")
        except openai.APIError as e:
            raise LLMGenerationError(f"OpenAI API error: {e}", original_exception=e)
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred during OpenAI generation: {e}", original_exception=e)

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Returns the current model and default parameters.

        Returns:
            Dict[str, Any]: A dictionary of model parameters.
        """
        return {"model": self.model, **self.default_params}

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string using a simple heuristic
        or an actual tokenizer if available. For a more accurate count,
        OpenAI recommends `tiktoken`. We'll implement a placeholder.

        Args:
            text (str): The text string to tokenize.

        Returns:
            int: The estimated number of tokens.
        """
        # A simple approximation. For production, consider using OpenAI's tiktoken library.
        # pip install tiktoken
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to a character-based estimate if tiktoken is not installed
            # This is a very rough estimate; actual token count will vary.
            return len(text.split()) * 4 // 3 # Rough estimate: 4 chars per token, or 3/4 word per token
        except Exception:
            # Fallback for models not supported by tiktoken or other issues
            return len(text.split()) * 4 // 3
