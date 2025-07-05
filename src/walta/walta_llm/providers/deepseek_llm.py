# src/walta/walta_llm/providers/deepseek_llm.py
import os
from typing import Dict, Any, List, Optional, Union
import json

from .base import LLMProviderProtocol, LLMGenerationError, ChatMessage

try:
    from llama_cpp import Llama
    _llama_cpp_available = True
except ImportError:
    _llama_cpp_available = False
    print("Warning: llama-cpp-python not installed. DeepseekLLMProvider will not function.")

class DeepseekLLMProvider(LLMProviderProtocol):
    """
    A concrete implementation of LLMProviderProtocol for running DeepSeek LLMs locally
    using llama-cpp-python.
    """
    def __init__(self, model_path: str = None, n_gpu_layers: int = 0, **kwargs: Any):
        if not _llama_cpp_available:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please install it with: pip install llama-cpp-python"
            )

        if model_path is None:
            model_path = os.getenv("DEEPSEEK_LLM_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "DeepSeek LLM model path not provided. "
                    "Please pass it as an argument or set the DEEPSEEK_LLM_MODEL_PATH environment variable."
                )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DeepSeek LLM model not found at: {model_path}")

        self.model_name = os.path.basename(model_path)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers # Number of layers to offload to GPU (-1 for all, 0 for CPU)
        self.default_params = kwargs

        try:
            print(f"Loading DeepSeek LLM from: {self.model_path} with {self.n_gpu_layers} GPU layers...")
            # DeepSeek models often use a specific chat_format in llama.cpp,
            # but llama-cpp-python should auto-detect from GGUF metadata or fall back to 'chatml' or 'llama-2'.
            # Explicitly setting chat_format='deepseek' might be needed for some older models or non-standard conversions.
            # We'll rely on auto-detection for now.
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=kwargs.get("n_ctx", 4096), # Default context window, adjust based on model
                n_batch=kwargs.get("n_batch", 512),
                n_gpu_layers=self.n_gpu_layers,
                verbose=kwargs.get("verbose_loading", False),
                # If chat template issues, uncomment and try: chat_format="deepseek"
            )
            print(f"Successfully loaded DeepSeek LLM: {self.model_name}")
        except Exception as e:
            raise LLMGenerationError(f"Failed to load DeepSeek LLM model from {model_path}: {e}", original_exception=e)

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        combined_params = {**self.default_params, **kwargs}
        try:
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=combined_params.get("max_tokens", 1000),
                temperature=combined_params.get("temperature", 0.7),
                top_p=combined_params.get("top_p", 1.0),
                top_k=combined_params.get("top_k", 40),
                stop=combined_params.get("stop_sequences", []),
                stream=False
            )
            return response["choices"][0]["text"]
        except Exception as e:
            raise LLMGenerationError(f"DeepSeek LLM text generation error: {e}", original_exception=e)

    def generate_chat_completion(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        combined_params = {**self.default_params, **kwargs}
        try:
            # DeepSeek models often use a specific chat template, e.g., <｜begin of sentence｜>User: {prompt} Assistant: <｜end of sentence｜>
            # llama-cpp-python's create_chat_completion should handle this automatically if the GGUF metadata is correct.
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            response = self.model.create_chat_completion(
                messages=formatted_messages,
                max_tokens=combined_params.get("max_tokens", 1000),
                temperature=combined_params.get("temperature", 0.7),
                top_p=combined_params.get("top_p", 1.0),
                stop=combined_params.get("stop_sequences", []),
                stream=False
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMGenerationError(f"DeepSeek LLM chat completion error: {e}", original_exception=e)

    def get_model_parameters(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            **self.default_params
        }

    def count_tokens(self, text: str) -> int:
        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception as e:
            raise LLMGenerationError(f"DeepSeek LLM token counting error: {e}", original_exception=e)

    def get_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.model.embed(text)
            if isinstance(embedding, list) and all(isinstance(x, float) for x in embedding):
                return embedding
            else:
                raise LLMGenerationError("DeepSeek LLM embedding response was malformed.")
        except Exception as e:
            raise LLMGenerationError(f"DeepSeek LLM embedding generation error: {e}", original_exception=e)

    def get_multimodal_analysis(self, text: str, image: Optional[bytes] = None) -> str:
        if image:
            raise LLMGenerationError(
                "DeepSeek LLM (llama-cpp-python) typically does not support direct image input "
                "for multimodal analysis with a standard text-based GGUF model. "
                "Consider a different model (e.g., LLaVA) and specific implementation."
            )
        elif text:
            return self.generate_text(f"Analyze the following: {text}")
        else:
            raise LLMGenerationError("No input (text or image) provided for multimodal analysis.")
