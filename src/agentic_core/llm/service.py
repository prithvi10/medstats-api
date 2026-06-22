"""
LLMService - Unified gateway for OpenAI API calls.
Supports text generation, structured outputs (Pydantic), and JSON mode.
"""
import os
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from src.shared.logging.logger import get_logger
from src.shared.config.settings import get_settings

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate unstructured text response."""
        model = model or settings.OPENAI_MODEL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await self.client.chat.completions.create(**kwargs)
        self._track_usage(response)
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        prompt: str,
        response_format: Type[T],
        system_prompt: str = "You are a helpful assistant. Respond with valid JSON.",
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output conforming to a Pydantic model."""
        model = model or settings.OPENAI_MODEL_MINI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        self._track_usage(response)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned unparseable response")
        return parsed

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Respond with valid JSON only.",
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate JSON-formatted text response."""
        model = model or settings.OPENAI_MODEL_MINI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        self._track_usage(response)
        return response.choices[0].message.content or "{}"

    def _track_usage(self, response):
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

    def get_usage_summary(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset_usage(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
