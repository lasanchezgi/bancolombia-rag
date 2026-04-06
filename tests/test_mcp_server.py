"""
Tests unitarios para el paquete mcp_server.

Verifica que create_server() devuelve una instancia válida de FastMCP,
que las tres tools están registradas, y que cada tool maneja correctamente
resultados, vacíos y errores de ChromaDB.

Usa unittest.mock para evitar conexiones reales a ChromaDB o OpenAI.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from src.mcp_server.server import create_server


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_tool_names(mcp: FastMCP) -> list[str]:
    """Devuelve los nombres de las tools registradas en un servidor FastMCP."""
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


def _call_tool(mcp: FastMCP, name: str, args: dict[str, Any]) -> str:
    """Llama una tool y devuelve su contenido como string (JSON serializado)."""
    result = asyncio.run(mcp.call_tool(name, args))
    # FastMCP 3.x devuelve ToolResult con .content[0].text (TextContent)
    return result.content[0].text  # type: ignore[union-attr]


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inyecta variables de entorno mínimas para evitar KeyError en las tools."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("CHROMA_HOST", "local")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("CHROMA_COLLECTION", "bancolombia_kb")


# ──────────────────────────────────────────────────────────────────────────────
# Tests: creación del servidor y registro de tools
# ──────────────────────────────────────────────────────────────────────────────


class TestMCPServer:
    def test_create_server_returns_fastmcp_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_server() debe devolver una instancia de FastMCP."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert isinstance(server, FastMCP)

    def test_server_name_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """El servidor debe tener el nombre 'BancolombiaKnowledgeBase'."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert server.name == "BancolombiaKnowledgeBase"

    def test_search_knowledge_base_tool_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La tool 'search_knowledge_base' debe estar registrada."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert "search_knowledge_base" in _get_tool_names(server)

    def test_get_article_by_url_tool_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La tool 'get_article_by_url' debe estar registrada."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert "get_article_by_url" in _get_tool_names(server)

    def test_list_categories_tool_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La tool 'list_categories' debe estar registrada."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert "list_categories" in _get_tool_names(server)

    def test_exactly_three_tools_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deben estar registradas exactamente 3 tools."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools.ChromaRepository"), patch("src.mcp_server.tools.Embedder"):
            server = create_server()
        assert len(_get_tool_names(server)) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Tests: comportamiento de las tools (con mocks de ChromaDB / Embedder)
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchKnowledgeBase:
    def _make_mock_repo(self) -> MagicMock:
        repo = MagicMock()
        repo.query.return_value = [
            {
                "text": "Texto de prueba",
                "url": "https://www.bancolombia.com/personas/cuentas",
                "title": "Cuentas",
                "category": "cuentas",
                "score": 0.92,
                "chunk_index": 0,
            },
        ]
        return repo

    def _make_mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1] * 1536]
        return embedder

    def _make_mcp(self) -> FastMCP:
        from src.mcp_server.tools import register_tools

        mcp = FastMCP("test")
        register_tools(mcp)
        return mcp

    def test_search_returns_required_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_knowledge_base debe retornar JSON con 'results', 'total' y 'query'."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools._get_repository", return_value=self._make_mock_repo()), patch("src.mcp_server.tools._get_embedder", return_value=self._make_mock_embedder()):
            content = _call_tool(self._make_mcp(), "search_knowledge_base", {"query": "cuentas de ahorro"})
        assert "results" in content
        assert "total" in content
        assert "query" in content

    def test_search_results_contain_expected_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cada resultado debe tener text, url, title, category, relevance_score."""
        _mock_env(monkeypatch)
        with patch("src.mcp_server.tools._get_repository", return_value=self._make_mock_repo()), patch("src.mcp_server.tools._get_embedder", return_value=self._make_mock_embedder()):
            content = _call_tool(self._make_mcp(), "search_knowledge_base", {"query": "cuentas"})
        assert "relevance_score" in content
        assert "url" in content
        assert "category" in content

    def test_search_empty_results_returns_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_knowledge_base con 0 resultados debe retornar mensaje informativo."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.query.return_value = []
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo), patch("src.mcp_server.tools._get_embedder", return_value=self._make_mock_embedder()):
            content = _call_tool(self._make_mcp(), "search_knowledge_base", {"query": "xyz123 inexistente"})
        assert "No se encontró" in content or '"total": 0' in content

    def test_search_handles_chroma_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_knowledge_base debe retornar 'error' si ChromaDB falla (sin lanzar excepción)."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.query.side_effect = ConnectionError("ChromaDB no disponible")
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo), patch("src.mcp_server.tools._get_embedder", return_value=self._make_mock_embedder()):
            content = _call_tool(self._make_mcp(), "search_knowledge_base", {"query": "test"})
        assert "error" in content.lower()


class TestGetArticleByUrl:
    def _make_mcp(self) -> FastMCP:
        from src.mcp_server.tools import register_tools

        mcp = FastMCP("test")
        register_tools(mcp)
        return mcp

    def test_get_article_returns_found_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_article_by_url debe retornar found=True cuando existe la URL."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.return_value = {
            "ids": ["chunk_0"],
            "documents": ["Texto del artículo completo"],
            "metadatas": [{"url": "https://www.bancolombia.com/personas/cuentas", "title": "Cuentas", "category": "cuentas", "chunk_index": 0, "extraction_date": "2024-01-01"}],
        }
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "get_article_by_url", {"url": "https://www.bancolombia.com/personas/cuentas"})
        assert '"found": true' in content or '"found":true' in content

    def test_get_article_returns_full_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_article_by_url debe incluir el campo 'full_text' con el texto concatenado."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.return_value = {
            "ids": ["c0", "c1"],
            "documents": ["Primer chunk.", "Segundo chunk."],
            "metadatas": [
                {"url": "https://example.com", "title": "T", "category": "c", "chunk_index": 0, "extraction_date": ""},
                {"url": "https://example.com", "title": "T", "category": "c", "chunk_index": 1, "extraction_date": ""},
            ],
        }
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "get_article_by_url", {"url": "https://example.com"})
        assert "full_text" in content
        assert "Primer chunk" in content

    def test_get_article_returns_found_false_for_missing_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_article_by_url debe retornar found=False para URL inexistente."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "get_article_by_url", {"url": "https://www.bancolombia.com/no-existe"})
        assert '"found": false' in content or "no encontrado" in content.lower() or "False" in content

    def test_get_article_handles_chroma_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_article_by_url debe retornar 'error' si ChromaDB falla."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.side_effect = ConnectionError("fallo de conexión")
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "get_article_by_url", {"url": "https://example.com"})
        assert "error" in content.lower()


class TestListCategories:
    def _make_mcp(self) -> FastMCP:
        from src.mcp_server.tools import register_tools

        mcp = FastMCP("test")
        register_tools(mcp)
        return mcp

    def test_list_categories_returns_required_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_categories debe retornar JSON con 'categories' y 'total_documents'."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.return_value = {
            "metadatas": [{"category": "cuentas"}, {"category": "cuentas"}, {"category": "creditos"}]
        }
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "list_categories", {})
        assert "categories" in content
        assert "total_documents" in content

    def test_list_categories_counts_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_categories debe contar documentos por categoría correctamente."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.return_value = {
            "metadatas": [{"category": "cuentas"}, {"category": "cuentas"}, {"category": "creditos"}]
        }
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "list_categories", {})
        assert "cuentas" in content
        assert "creditos" in content
        assert "total_documents" in content
        assert "3" in content

    def test_list_categories_handles_chroma_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_categories debe retornar 'error' si ChromaDB falla."""
        _mock_env(monkeypatch)
        mock_repo = MagicMock()
        mock_repo.collection.get.side_effect = RuntimeError("ChromaDB error")
        with patch("src.mcp_server.tools._get_repository", return_value=mock_repo):
            content = _call_tool(self._make_mcp(), "list_categories", {})
        assert "error" in content.lower()
