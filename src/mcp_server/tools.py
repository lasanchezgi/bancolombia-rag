"""
Definición de herramientas MCP para el pipeline de retrieval RAG.

Cada función decorada con @mcp.tool se convierte en una herramienta invocable
por cualquier cliente MCP compatible (e.g. el agente OpenAI). Las herramientas
encapsulan las operaciones de búsqueda y recuperación sobre el vector store.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP


def register_tools(mcp: FastMCP) -> None:
    """Registra todas las herramientas de retrieval en la instancia FastMCP.

    Las herramientas se definen como funciones internas decoradas con
    ``@mcp.tool`` para que FastMCP las descubra y exponga automáticamente.

    Args:
        mcp: Instancia de FastMCP sobre la que se registran las herramientas.
    """

    @mcp.tool  # type: ignore[misc]
    def search_knowledge_base(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Busca artículos en la base de conocimiento relevantes para una consulta.

        Genera el embedding de la consulta y recupera los ``top_k`` chunks
        más similares almacenados en ChromaDB.

        Args:
            query: Pregunta o términos de búsqueda en lenguaje natural.
            top_k: Número de resultados a devolver (default: 5).

        Returns:
            Lista de dicts con ``document`` (texto), ``metadata`` y ``score``.
        """
        raise NotImplementedError

    @mcp.tool  # type: ignore[misc]
    def get_article_by_url(url: str) -> dict[str, Any]:
        """Recupera el artículo indexado correspondiente a una URL específica.

        Args:
            url: URL exacta del artículo tal como fue indexada durante el scraping.

        Returns:
            Dict con ``url``, ``title``, ``document`` y ``metadata`` del artículo.

        Raises:
            KeyError: Si no existe ningún artículo indexado con esa URL.
        """
        raise NotImplementedError

    @mcp.tool  # type: ignore[misc]
    def list_categories() -> list[str]:
        """Lista todas las categorías temáticas disponibles en la base de conocimiento.

        Las categorías se derivan de los metadatos almacenados durante la indexación
        y permiten al agente orientar búsquedas dentro de un dominio específico.

        Returns:
            Lista de strings con los nombres de las categorías únicas indexadas.
        """
        raise NotImplementedError
