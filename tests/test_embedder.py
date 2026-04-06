"""
Tests unitarios para el Embedder.

Usa unittest.mock para simular las llamadas a la API de OpenAI, sin realizar
peticiones de red reales. Cubre batching, enriquecimiento de chunks, y
reintentos ante RateLimitError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError

from src.embeddings.embedder import Embedder, _BATCH_SIZE

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_embedding_response(texts: list[str]) -> MagicMock:
    """Crea un mock de respuesta de la API de OpenAI Embeddings."""
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]
    return response


def _make_embedder() -> Embedder:
    """Instancia un Embedder con clave de API falsa (no realiza llamadas reales)."""
    return Embedder(api_key="sk-test-fake-key")


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbedTexts:
    def test_embed_texts_returns_list_of_vectors(self) -> None:
        """embed_texts() debe devolver una lista de vectores (list[list[float]])."""
        embedder = _make_embedder()
        texts = ["texto uno", "texto dos", "texto tres"]

        with patch.object(embedder.client.embeddings, "create", side_effect=lambda **kw: _make_embedding_response(kw["input"])):
            result = embedder.embed_texts(texts)

        assert isinstance(result, list)
        assert len(result) == len(texts)
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(f, float) for f in result[0])

    def test_embed_texts_returns_correct_count(self) -> None:
        """embed_texts() debe devolver exactamente len(texts) vectores."""
        embedder = _make_embedder()
        texts = [f"texto {i}" for i in range(5)]

        with patch.object(embedder.client.embeddings, "create", side_effect=lambda **kw: _make_embedding_response(kw["input"])):
            result = embedder.embed_texts(texts)

        assert len(result) == 5

    def test_embed_texts_processes_in_batches(self) -> None:
        """Con 250 textos debe hacer ceil(250/100) = 3 llamadas a la API."""
        embedder = _make_embedder()
        texts = [f"texto {i}" for i in range(250)]

        mock_create = MagicMock(side_effect=lambda **kw: _make_embedding_response(kw["input"]))

        with patch.object(embedder.client.embeddings, "create", mock_create):
            result = embedder.embed_texts(texts)

        assert mock_create.call_count == 3  # ceil(250 / 100)
        assert len(result) == 250

    def test_embed_texts_single_batch_for_small_input(self) -> None:
        """Con menos de 100 textos debe hacer exactamente 1 llamada a la API."""
        embedder = _make_embedder()
        texts = [f"texto {i}" for i in range(50)]

        mock_create = MagicMock(side_effect=lambda **kw: _make_embedding_response(kw["input"]))

        with patch.object(embedder.client.embeddings, "create", mock_create):
            embedder.embed_texts(texts)

        assert mock_create.call_count == 1

    def test_embed_texts_retries_on_rate_limit(self) -> None:
        """ante RateLimitError debe reintentar hasta _MAX_RETRIES veces."""
        embedder = _make_embedder()
        texts = ["texto de prueba"]

        # Falla 2 veces con RateLimitError, luego tiene éxito
        call_count = {"n": 0}

        def side_effect(**kw):
            call_count["n"] += 1
            if call_count["n"] < 3:
                mock_req = MagicMock()
                mock_res = MagicMock()
                mock_res.status_code = 429
                raise RateLimitError("rate limit", response=mock_res, body={})
            return _make_embedding_response(kw["input"])

        with patch.object(embedder.client.embeddings, "create", side_effect=side_effect):
            with patch("src.embeddings.embedder.time.sleep"):  # evitar espera real
                result = embedder.embed_texts(texts)

        assert len(result) == 1
        assert call_count["n"] == 3  # 2 fallos + 1 éxito


class TestEmbedChunks:
    def test_embed_chunks_adds_embedding_field(self) -> None:
        """embed_chunks() debe agregar el campo 'embedding' a cada chunk."""
        embedder = _make_embedder()
        chunks = [
            {"chunk_id": "c_0", "text": "texto uno", "url": "https://example.com"},
            {"chunk_id": "c_1", "text": "texto dos", "url": "https://example.com"},
        ]

        with patch.object(embedder.client.embeddings, "create", side_effect=lambda **kw: _make_embedding_response(kw["input"])):
            result = embedder.embed_chunks(chunks)

        assert len(result) == 2
        for chunk in result:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) == 1536

    def test_embed_chunks_preserves_original_fields(self) -> None:
        """embed_chunks() no debe eliminar los campos originales de cada chunk."""
        embedder = _make_embedder()
        chunks = [{"chunk_id": "abc_0", "text": "contenido", "url": "https://x.com", "category": "cuentas"}]

        with patch.object(embedder.client.embeddings, "create", side_effect=lambda **kw: _make_embedding_response(kw["input"])):
            result = embedder.embed_chunks(chunks)

        assert result[0]["chunk_id"] == "abc_0"
        assert result[0]["url"] == "https://x.com"
        assert result[0]["category"] == "cuentas"

    def test_embed_chunks_empty_input_returns_empty(self) -> None:
        """embed_chunks([]) debe devolver []."""
        embedder = _make_embedder()

        with patch.object(embedder.client.embeddings, "create", side_effect=lambda **kw: _make_embedding_response(kw["input"])):
            result = embedder.embed_chunks([])

        assert result == []
