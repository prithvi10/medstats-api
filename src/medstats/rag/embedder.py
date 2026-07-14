"""Batch embedding via OpenAI text-embedding-3-small."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from openai import AsyncOpenAI

from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# text-embedding-3-small limit is 8192 tokens; ~4 chars/token for dense text
MAX_EMBEDDING_CHARS = 24_000


def _truncate_for_embedding(text: str) -> str:
    if len(text) <= MAX_EMBEDDING_CHARS:
        return text
    logger.warning(
        "Truncating oversized embedding input",
        original_chars=len(text),
        max_chars=MAX_EMBEDDING_CHARS,
    )
    return text[:MAX_EMBEDDING_CHARS]


class ChunkEmbedder:
    """Embed text chunks in batches with rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        import asyncio

        loop_id = id(asyncio.get_running_loop())
        if self._client is None or getattr(self, "_client_loop_id", None) != loop_id:
            import httpx

            http_client = httpx.AsyncClient(
                verify=False,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            )
            self._client = AsyncOpenAI(api_key=self.api_key, http_client=http_client)
            self._client_loop_id = loop_id
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # OpenAI rejects empty strings and inputs over the token limit
            batch = [_truncate_for_embedding(t if t.strip() else " ") for t in batch]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_emb = [item.embedding for item in response.data]
            all_embeddings.extend(batch_emb)
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        result = await self.embed_texts([query])
        return result[0] if result else []
