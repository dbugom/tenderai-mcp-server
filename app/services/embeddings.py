"""Embedding service — Voyage AI wrapper for vector embeddings."""

from __future__ import annotations

import logging
from typing import Optional

import voyageai

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Async wrapper around Voyage AI for generating text embeddings."""

    def __init__(self, api_key: str, model: str = "voyage-3-lite", dimensions: int = 1024):
        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        result = await self.client.embed(
            texts=[text],
            model=self.model,
            input_type="document",
        )
        vector = result.embeddings[0]
        logger.debug("Generated embedding: model=%s, dims=%d", self.model, len(vector))
        return vector

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query.

        Uses input_type="query" for asymmetric retrieval (query vs document).

        Args:
            query: The search query text.

        Returns:
            List of floats representing the query embedding vector.
        """
        result = await self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
        )
        return result.embeddings[0]

    async def embed_batch(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.
            input_type: Either "document" or "query".

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        result = await self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
        )
        logger.debug("Batch embedded %d texts: model=%s", len(texts), self.model)
        return result.embeddings
