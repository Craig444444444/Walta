"""
Google Gemini LLM Provider Implementation.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from . import LLMProviderProtocol, ProviderError

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProviderProtocol):
    """Google Gemini implementation of LLM provider."""
    
    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.embedding_model = genai.GenerativeModel("embedding-001")
        
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY environment variable not set")
        logger.info(f"GeminiProvider initialized with model {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector using Gemini."""
        try:
            response = await self.embedding_model.embed_content(
                model="embedding-001",
                content=text,
            )
            logger.debug(f"Generated embedding of dimension {len(response.embedding)}")
            return response.embedding
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise ProviderError(f"Gemini embedding failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Get completion from Gemini."""
        try:
            response = await self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            completion = response.text
            logger.debug(f"Generated completion of length {len(completion)}")
            return completion
        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
            raise ProviderError(f"Gemini completion failed: {e}")

    async def get_multimodal_analysis(
        self,
        text: str,
        image: Optional[bytes] = None
    ) -> str:
        """Get multimodal analysis using Gemini Pro Vision."""
        try:
            if image is None:
                return await self.get_completion(text)

            vision_model = genai.GenerativeModel("gemini-pro-vision")
            response = await vision_model.generate_content(
                [text, {"mime_type": "image/jpeg", "data": image}]
            )
            analysis = response.text
            logger.debug(f"Generated multimodal analysis of length {len(analysis)}")
            return analysis
        except Exception as e:
            logger.error(f"Gemini multimodal analysis failed: {e}")
            raise ProviderError(f"Gemini multimodal analysis failed: {e}")
