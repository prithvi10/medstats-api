"""
LLMService — Unified gateway for OpenAI API calls.

Supports:
  - Text generation (narrative, summaries)
  - Structured outputs (Pydantic models via response_format)
  - JSON mode (review checks, classification)

Cost tracking included for per-submission billing.
"""
import json
from typing import Type, TypeVar, Optional, Dict, Any

from pydantic import BaseModel
from openai import AsyncOpenAI

from src.shared.logging.logger import get_logger
from src.shared.config.settings import get_settings

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """Async OpenAI client wrapper with automatic model compatibility handling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env or pass directly."
            )
        import httpx
        http_client = httpx.AsyncClient(verify=False)
        self.client = AsyncOpenAI(api_key=self.api_key, http_client=http_client)
        self.default_model = model or settings.OPENAI_MODEL
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def _supports_temperature(self, model: str) -> bool:
        """GPT-5+ and o-series only support default temperature."""
        return not any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate unstructured text response."""
        model = model or self.default_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if self._supports_temperature(model):
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens

        logger.debug("LLM call", model=model, prompt_len=len(prompt))
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

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
        }
        if self._supports_temperature(model):
            kwargs["temperature"] = temperature

        logger.debug("LLM structured call", model=model, schema=response_format.__name__)
        response = await self.client.beta.chat.completions.parse(**kwargs)
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
    ) -> Dict[str, Any]:
        """Generate JSON response and parse it into a dict."""
        model = model or settings.OPENAI_MODEL_MINI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._supports_temperature(model):
            kwargs["temperature"] = temperature

        logger.debug("LLM JSON call", model=model)
        response = await self.client.chat.completions.create(**kwargs)
        self._track_usage(response)
        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, returning empty", raw=raw[:200])
            return {}

    def _track_usage(self, response):
        """Track token consumption for cost monitoring."""
        self.call_count += 1
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

    def get_usage_summary(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "call_count": self.call_count,
            "model": self.default_model,
        }

    def reset_usage(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0


def get_llm_service() -> Optional[LLMService]:
    """
    Factory function — returns LLMService if API key is configured, else None.
    Agents gracefully degrade to template-based generation when None.
    """
    if not settings.OPENAI_API_KEY:
        logger.info("No OPENAI_API_KEY configured — LLM features disabled")
        return None
    return LLMService()
