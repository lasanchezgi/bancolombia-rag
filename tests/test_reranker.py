"""
Tests unitarios para Reranker y la integración en search_knowledge_base.

Usa unittest.mock para evitar cargar el modelo Cross-Encoder real
(no requiere descarga de internet, apto para CI).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_docs(n: int) -> list[dict[str, Any]]:
    """Genera n documentos de prueba con los campos mínimos."""
    return [
        {
            "text": f"Documento {i}",
            "url": f"https://bancolombia.com/{i}",
            "title": f"Título {i}",
            "category": "cuentas",
            "score": 0.9 - i * 0.1,
            "chunk_index": i,
        }
        for i in range(n)
    ]


def _make_mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1] * 1536]
    return embedder


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("CHROMA_HOST", "local")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("CHROMA_COLLECTION", "bancolombia_kb")


def _call_tool(mcp: FastMCP, name: str, args: dict[str, Any]) -> str:
    result = asyncio.run(mcp.call_tool(name, args))
    return result.content[0].text  # type: ignore[union-attr]


def _make_mcp() -> FastMCP:
    from src.mcp_server.tools import register_tools

    mcp = FastMCP("test")
    register_tools(mcp)
    return mcp


# ──────────────────────────────────────────────────────────────────────────────
# Tests: clase Reranker (unit)
# ──────────────────────────────────────────────────────────────────────────────


class TestReranker:
    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_returns_top_k_results(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() debe retornar exactamente top_k documentos."""
        mock_cross_encoder.return_value.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]

        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        docs = _make_docs(5)
        result = reranker.rerank("query", docs, top_k=3)

        assert len(result) == 3

    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_orders_by_score_descending(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() debe retornar documentos ordenados de mayor a menor score."""
        mock_cross_encoder.return_value.predict.return_value = [0.3, 0.9, 0.1, 0.7, 0.5]

        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        docs = _make_docs(5)
        result = reranker.rerank("query", docs, top_k=3)

        assert result[0]["rerank_score"] == pytest.approx(0.9)
        assert result[1]["rerank_score"] == pytest.approx(0.7)
        assert result[2]["rerank_score"] == pytest.approx(0.5)

    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_adds_rerank_score_field(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() debe agregar el campo 'rerank_score' a cada documento retornado."""
        mock_cross_encoder.return_value.predict.return_value = [0.5, 0.8, 0.2]

        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        docs = _make_docs(3)
        result = reranker.rerank("query", docs, top_k=3)

        for doc in result:
            assert "rerank_score" in doc
            assert isinstance(doc["rerank_score"], float)

    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_empty_documents_returns_empty(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() con lista vacía debe retornar lista vacía sin llamar al modelo."""
        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        result = reranker.rerank("query", [], top_k=5)

        assert result == []
        mock_cross_encoder.return_value.predict.assert_not_called()

    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_top_k_larger_than_docs(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() con top_k mayor que docs disponibles debe retornar todos los docs sin error."""
        mock_cross_encoder.return_value.predict.return_value = [0.5, 0.8, 0.2]

        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        docs = _make_docs(3)
        result = reranker.rerank("query", docs, top_k=10)

        assert len(result) == 3

    @patch("src.embeddings.reranker.CrossEncoder")
    def test_rerank_fallback_on_model_error(self, mock_cross_encoder: MagicMock) -> None:
        """rerank() debe hacer fallback graceful si el modelo lanza excepción."""
        mock_cross_encoder.return_value.predict.side_effect = RuntimeError("modelo no disponible")

        from src.embeddings.reranker import Reranker

        reranker = Reranker()
        docs = _make_docs(3)

        # No debe lanzar excepción
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        # En fallback, rerank_score es None
        for doc in result:
            assert doc["rerank_score"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: integración reranking en search_knowledge_base
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchKnowledgeBaseReranking:
    def _make_mock_repo(self, n_results: int = 5) -> MagicMock:
        repo = MagicMock()
        repo.query.return_value = _make_docs(n_results)
        return repo

    def test_search_knowledge_base_with_reranking_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Con use_reranking=True, el reranker debe ser llamado y ChromaDB recibe top_k*3."""
        _mock_env(monkeypatch)
        mock_repo = self._make_mock_repo(n_results=15)
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = _make_docs(5)

        # Resetear singleton para que use nuestro mock
        import src.mcp_server.tools as tools_module

        tools_module._reranker_instance = None

        with (
            patch("src.mcp_server.tools._get_repository", return_value=mock_repo),
            patch("src.mcp_server.tools._get_embedder", return_value=_make_mock_embedder()),
            patch("src.mcp_server.tools.get_reranker", return_value=mock_reranker),
        ):
            content = _call_tool(_make_mcp(), "search_knowledge_base", {"query": "cuentas", "use_reranking": True})

        # Reranker debe haber sido llamado
        mock_reranker.rerank.assert_called_once()

        # ChromaDB debe haber recibido top_k * 3 = 15 (top_k=5 default, min(15,30)=15)
        call_kwargs = mock_repo.query.call_args
        assert call_kwargs.kwargs["top_k"] == 15

        # El resultado debe indicar el método correcto
        data = json.loads(content)
        assert data["results"][0]["retrieval_method"] == "chromadb+reranking"

    def test_search_knowledge_base_with_reranking_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Con use_reranking=False, el reranker NO debe ser llamado y ChromaDB recibe top_k exacto."""
        _mock_env(monkeypatch)
        mock_repo = self._make_mock_repo(n_results=5)
        mock_reranker = MagicMock()

        import src.mcp_server.tools as tools_module

        tools_module._reranker_instance = None

        with (
            patch("src.mcp_server.tools._get_repository", return_value=mock_repo),
            patch("src.mcp_server.tools._get_embedder", return_value=_make_mock_embedder()),
            patch("src.mcp_server.tools.get_reranker", return_value=mock_reranker),
        ):
            content = _call_tool(_make_mcp(), "search_knowledge_base", {"query": "cuentas", "use_reranking": False})

        # Reranker NO debe haber sido llamado
        mock_reranker.rerank.assert_not_called()

        # ChromaDB debe haber recibido top_k exacto = 5
        call_kwargs = mock_repo.query.call_args
        assert call_kwargs.kwargs["top_k"] == 5

        # El resultado debe indicar el método correcto
        data = json.loads(content)
        assert data["results"][0]["retrieval_method"] == "chromadb_only"
