# src/walta/walta_llm/llm_manager.py
"""
Walta LLM Manager - High-level interface for LLM operations.
"""

import time
import asyncio
import logging # Explicitly import logging here
from typing import Any, Dict, Optional, List # Added List for ChatMessage types
from datetime import datetime

# Assuming LLMProviderProtocol and LLMGenerationError are available via __init__.py
from .providers import LLMProviderProtocol, LLMGenerationError # Added LLMGenerationError
from .providers.openai import OpenAIProvider
from .providers.gemini import GeminiProvider
from .providers.base import ChatMessage # Import ChatMessage type from base

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider: str = "gemini", model: str = None) -> LLMProviderProtocol:
        """Create an LLM provider instance."""
        if provider.lower() == "openai":
            # Pass kwargs directly to provider constructor if needed, e.g., api_key
            return OpenAIProvider(model=model or "gpt-4")
        elif provider.lower() == "gemini":
            return GeminiProvider(model=model or "gemini-pro")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

class WaltaLLM:
    """High-level interface for Walta's LLM operations."""
    
    def __init__(
        self,
        primary_provider_name: str = "gemini",
        fallback_provider_name: str = "openai",
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None
    ):
        self.primary_provider_name = primary_provider_name
        self.fallback_provider_name = fallback_provider_name
        self.primary = ModelFactory.create_provider(primary_provider_name, primary_model)
        self.fallback = ModelFactory.create_provider(fallback_provider_name, fallback_model)
        
        logger.info(
            f"WaltaLLM initialized with primary: {primary_provider_name}, "
            f"fallback: {fallback_provider_name}"
        )

    async def analyze(
        self,
        text: str,
        image: Optional[bytes] = None,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """Perform analysis with fallback support."""
        start_time = time.time()
        provider_used = self.primary_provider_name
        result_content = ""
        
        try:
            # Calls get_multimodal_analysis which is now in LLMProviderProtocol
            result_content = await self.primary.get_multimodal_analysis(text, image)
        except LLMGenerationError as e: # Catch specific LLMGenerationError
            logger.warning(
                f"Primary LLM provider ({self.primary_provider_name}) failed: {e}"
            )
            if use_fallback:
                provider_used = self.fallback_provider_name
                try:
                    # Calls get_multimodal_analysis which is now in LLMProviderProtocol
                    result_content = await self.fallback.get_multimodal_analysis(
                        text, image
                    )
                except LLMGenerationError as fallback_e: # Catch specific LLMGenerationError
                    logger.error(
                        f"Fallback LLM provider ({self.fallback_provider_name}) "
                        f"also failed: {fallback_e}"
                    )
                    raise RuntimeError(
                        "Both primary and fallback LLM providers failed."
                    ) from fallback_e
            else:
                raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred with {self.primary_provider_name}: {e}")
            raise

        return {
            "result": result_content,
            "provider": provider_used,
            "latency": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": f"req_{int(time.time()*1000)}",
            "metadata": {
                "text_length": len(text),
                "has_image": image is not None,
                "fallback_enabled": use_fallback
            }
        }

    async def get_embedding(
        self,
        text: str, # Changed type hint to str for consistency as per get_embedding signature in WaltaLLM
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """Get embeddings with fallback support."""
        start_time = time.time()
        provider_used = self.primary_provider_name
        vector_data = []
        
        try:
            # Calls get_embedding which is now in LLMProviderProtocol
            vector_data = await self.primary.get_embedding(text)
        except LLMGenerationError as e: # Catch specific LLMGenerationError
            logger.warning(
                f"Primary LLM provider ({self.primary_provider_name}) "
                f"embedding failed: {e}"
            )
            if use_fallback:
                provider_used = self.fallback_provider_name
                try:
                    # Calls get_embedding which is now in LLMProviderProtocol
                    vector_data = await self.fallback.get_embedding(text)
                except LLMGenerationError as fallback_e: # Catch specific LLMGenerationError
                    logger.error(
                        f"Fallback LLM provider ({self.fallback_provider_name}) "
                        f"embedding also failed: {fallback_e}"
                    )
                    raise RuntimeError(
                        "Both primary and fallback LLM providers failed to get embedding."
                    ) from fallback_e
            else:
                raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred with {self.primary_provider_name} during embedding: {e}")
            raise

        return {
            "vector": vector_data,
            "provider": provider_used,
            "latency": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": f"emb_{int(time.time()*1000)}",
            "metadata": {
                "text_length": len(text),
                "vector_dimension": len(vector_data), # This assumes vector_data is List[float] or List[List[float]]
                "fallback_enabled": use_fallback
            }
        }
