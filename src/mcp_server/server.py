"""
Punto de entrada del servidor FastMCP para bancolombia-rag.

Instancia la aplicación FastMCP, delega el registro de herramientas
a tools.register_tools() y arranca el servidor en MCP_SERVER_HOST:MCP_SERVER_PORT.

Puede ejecutarse directamente como módulo:
    uv run python -m src.mcp_server.server
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

from .tools import register_tools


def create_server() -> FastMCP:
    """Construye y configura la instancia del servidor FastMCP.

    Crea el servidor, registra las herramientas de retrieval (search_knowledge_base,
    get_article_by_url, list_categories) y el resource knowledgebase://stats,
    y devuelve la instancia lista para iniciar la escucha.

    Returns:
        Instancia de FastMCP completamente configurada.
    """
    mcp: FastMCP = FastMCP("bancolombia-rag")

    register_tools(mcp)

    @mcp.resource("knowledgebase://stats")  # type: ignore[misc]
    def knowledgebase_stats() -> dict[str, Any]:
        """Devuelve estadísticas agregadas de la base de conocimiento indexada.

        Returns:
            Dict con métricas como ``total_documents``, ``total_categories``
            y ``last_indexed_at``.
        """
        raise NotImplementedError

    return mcp


if __name__ == "__main__":
    server = create_server()
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8080"))
    server.run(host=host, port=port)
