"""
Embedder que genera vectores de texto usando la API de OpenAI.

Soporta procesamiento en batches de 100 textos y reintentos automáticos
ante errores de rate limit o de la API, con backoff exponencial.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import APIError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


class Embedder:
    """Genera embeddings para listas de textos usando la API de OpenAI.

    Attributes:
        model: Identificador del modelo de embeddings de OpenAI.
        client: Instancia del cliente OpenAI.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """Inicializa el Embedder con una clave de API explícita.

        Args:
            api_key: Clave de API de OpenAI.
            model: Identificador del modelo de embeddings a usar.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos en batches de 100.

        Args:
            texts: Lista de strings a convertir en vectores.

        Returns:
            Lista de vectores de embedding, uno por cada texto. El orden
            se preserva (``result[i]`` corresponde a ``texts[i]``).

        Raises:
            openai.APIError: Si la llamada a la API falla tras 3 reintentos.
        """
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(texts), _BATCH_SIZE), start=1):
            batch = texts[start : start + _BATCH_SIZE]
            logger.info("Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch))

            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Agrega el campo ``"embedding"`` a cada chunk.

        Args:
            chunks: Lista de dicts de chunks (salida del Chunker).

        Returns:
            Mismos dicts enriquecidos con ``"embedding": list[float]``.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)
        return [{**c, "embedding": e} for c, e in zip(chunks, embeddings)]

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Llama a la API de embeddings con reintentos ante rate limit / errores.

        Args:
            batch: Lista de textos (máx. 100) a embeddear en una sola llamada.

        Returns:
            Lista de vectores de embedding para el batch.

        Raises:
            openai.APIError: Si el error persiste tras _MAX_RETRIES intentos.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                return [item.embedding for item in response.data]
            except (RateLimitError, APIError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "OpenAI error (attempt %d/%d): %s - retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        exc,
                        _RETRY_BACKOFF,
                    )
                    time.sleep(_RETRY_BACKOFF)

        raise last_exc  # type: ignore[misc]
