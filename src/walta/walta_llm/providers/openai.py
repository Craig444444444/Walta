"""
OpenAI LLM Provider Implementation.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from . import LLMProviderProtocol, ProviderError

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProviderProtocol):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
        logger.info(f"OpenAIProvider initialized with model {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        try:
            response = await openai.Embedding.acreate(
                input=text,
                model="text-embedding-ada-002"
            )
            logger.debug(f"Generated embedding of dimension {len(response['data'][0]['embedding'])}")
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise ProviderError(f"OpenAI embedding failed: {e}")

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
        """Get completion from OpenAI."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            completion = response.choices[0].message.content
            logger.debug(f"Generated completion of length {len(completion)}")
            return completion
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise ProviderError(f"OpenAI completion failed: {e}")

    async def get_multimodal_analysis(
        self,
        text: str,
        image: Optional[bytes] = None
    ) -> str:
        """Get multimodal analysis from GPT-4V."""
        try:
            if image is None:
                return await self.get_completion(text)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image.decode()}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            analysis = response.choices[0].message.content
            logger.debug(f"Generated multimodal analysis of length {len(analysis)}")
            return analysis
        except Exception as e:
            logger.error(f"OpenAI multimodal analysis failed: {e}")
            raise ProviderError(f"OpenAI multimodal analysis failed: {e}")
