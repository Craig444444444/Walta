# src/walta/walta_llm/providers/gemini.py
import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional, Union

# Import from the base module within the same package
from .base import LLMProviderProtocol, LLMGenerationError, ChatMessage

class GeminiProvider(LLMProviderProtocol):
    """
    A concrete implementation of LLMProviderProtocol for Google Gemini API.
    """
    def __init__(self, api_key: str = None, model: str = "gemini-pro", **kwargs: Any):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Please pass it as an argument or set the GEMINI_API_KEY environment variable."
                )
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model_name=self.model_name)
        self.default_params = kwargs

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text using Google Gemini's model.

        Args:
            prompt (str): The input prompt.
            **kwargs: Overrides for default model parameters (e.g., temperature, max_output_tokens, stop_sequences).

        Returns:
            str: The generated text.

        Raises:
            LLMGenerationError: If the API call fails.
        """
        combined_params = {**self.default_params, **kwargs}
        generation_config = genai.GenerationConfig(
            temperature=combined_params.get("temperature", 0.7),
            max_output_tokens=combined_params.get("max_tokens", 1000), # Gemini uses max_output_tokens
            stop_sequences=combined_params.get("stop_sequences", []),
            top_p=combined_params.get("top_p", 1.0),
            top_k=combined_params.get("top_k", 0)
        )
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=generation_config
            )
            return response.text
        except genai.APIError as e:
            raise LLMGenerationError(f"Gemini API error: {e}", original_exception=e)
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred during Gemini generation: {e}", original_exception=e)

    def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """
        Generates a chat completion using Google Gemini's chat capabilities.

        Args:
            messages (List[ChatMessage]): A list of chat messages.
            **kwargs: Overrides for default model parameters.

        Returns:
            str: The generated response.

        Raises:
            LLMGenerationError: If the API call fails.
        """
        combined_params = {**self.default_params, **kwargs}
        generation_config = genai.GenerationConfig(
            temperature=combined_params.get("temperature", 0.7),
            max_output_tokens=combined_params.get("max_tokens", 1000),
            stop_sequences=combined_params.get("stop_sequences", []),
            top_p=combined_params.get("top_p", 1.0),
            top_k=combined_params.get("top_k", 0)
        )
        try:
            # Gemini's chat history format expects roles like 'user' and 'model'
            # Adjusting input messages to fit Gemini's expected format if necessary
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    formatted_messages.append({"role": "model", "parts": [msg["content"]]})
                else:
                    formatted_messages.append({"role": msg["role"], "parts": [msg["content"]]})

            chat_session = self.model.start_chat(history=formatted_messages[:-1])
            response = chat_session.send_message(formatted_messages[-1]["parts"][0], generation_config=generation_config)
            return response.text
        except genai.APIError as e:
            raise LLMGenerationError(f"Gemini API error: {e}", original_exception=e)
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred during Gemini chat completion: {e}", original_exception=e)

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Returns the current model and default parameters.

        Returns:
            Dict[str, Any]: A dictionary of model parameters.
        """
        return {"model": self.model_name, **self.default_params}

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string using Gemini's API.

        Args:
            text (str): The text string to tokenize.

        Returns:
            int: The number of tokens in the text.

        Raises:
            LLMGenerationError: If token counting fails.
        """
        try:
            response = self.model.count_tokens(contents=[text])
            return response.total_tokens
        except genai.APIError as e:
            raise LLMGenerationError(f"Gemini API error during token counting: {e}", original_exception=e)
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred during Gemini token counting: {e}", original_exception=e)

    def get_multimodal_analysis(self, text: str, image: Optional[bytes] = None) -> str:
        """
        Performs multimodal analysis (text and optional image) using Gemini's capabilities.
        Note: This requires a Gemini model capable of multimodal input (e.g., 'gemini-pro-vision').

        Args:
            text (str): The text prompt.
            image (Optional[bytes]): The image data as bytes.

        Returns:
            str: The analysis result.

        Raises:
            LLMGenerationError: If the API call fails or model does not support multimodal.
        """
        if not image and text:
            # If only text, defer to generate_text
            return self.generate_text(text)
        elif not image:
            raise LLMGenerationError("Image is required for multimodal analysis if text is empty.")
        
        # This part assumes a model like 'gemini-pro-vision' is used for multimodal
        # The 'gemini-pro' model specified in __init__ does not support image input directly.
        # You would need to instantiate a different model for this if the base model is not multimodal.
        # For this example, we'll assume the self.model is set up for it, or handle specific model.
        
        try:
            from PIL import Image
            import io
            
            # Load the image from bytes
            img = Image.open(io.BytesIO(image))
            
            # Gemini's generate_content can take a list of parts, including text and images
            contents = [text, img] if text else [img]
            
            response = self.model.generate_content(contents=contents)
            return response.text
        except genai.APIError as e:
            raise LLMGenerationError(f"Gemini API error during multimodal analysis: {e}", original_exception=e)
        except ImportError:
            raise LLMGenerationError("Pillow (PIL) library is required for image processing in multimodal analysis. Please install it (`pip install Pillow`).")
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred during Gemini multimodal analysis: {e}", original_exception=e)
