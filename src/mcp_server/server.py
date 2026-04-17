"""
Punto de entrada del servidor FastMCP para bancolombia-rag.

Instancia la aplicación FastMCP, registra las herramientas de retrieval
y soporta transporte dual: stdio (para el agente interno) y SSE (para
clientes externos como Claude Desktop).

Uso:
    uv run python src/mcp_server/server.py              # stdio (default)
    MCP_TRANSPORT=sse uv run python src/mcp_server/server.py  # SSE
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

try:
    from .tools import preload_reranker, register_tools
except ImportError:
    from src.mcp_server.tools import preload_reranker, register_tools  # type: ignore[no-redef]


def create_server() -> FastMCP:
    """Construye y configura la instancia del servidor FastMCP.

    Crea el servidor con instrucciones para el agente, registra las
    herramientas de retrieval y el resource de estadísticas.

    Returns:
        Instancia de FastMCP completamente configurada.
    """
    load_dotenv()

    mcp: FastMCP = FastMCP(
        name="BancolombiaKnowledgeBase",
        instructions=(
            "Servidor MCP que expone la base de conocimiento del sitio web de Bancolombia personas. "
            "Contiene información sobre cuentas, tarjetas de crédito, tarjetas débito, créditos, "
            "beneficios y giros internacionales. "
            "Usa search_knowledge_base para buscar información relevante, "
            "get_article_by_url para obtener contenido completo de una página específica, "
            "y list_categories para ver qué temas están disponibles."
        ),
    )

    register_tools(mcp)
    preload_reranker()
    return mcp


if __name__ == "__main__":
    load_dotenv()

    transport = os.getenv("MCP_TRANSPORT", "stdio")
    server = create_server()

    if transport == "sse":
        host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_SERVER_PORT", "8000"))
        logger.info("Starting MCP server SSE on %s:%s", host, port)
        server.run(transport="sse", host=host, port=port)
    else:
        server.run(transport="stdio")
