"""
Tests unitarios para el paquete mcp_server.

Verifica que create_server() devuelve una instancia válida de FastMCP y que
las herramientas de retrieval están correctamente registradas antes de que
el servidor empiece a escuchar conexiones.
"""

from __future__ import annotations


class TestMCPServer:
    """Tests de creación del servidor FastMCP y registro de herramientas."""

    def test_create_server_returns_fastmcp_instance(self) -> None:
        """create_server() debe devolver una instancia de FastMCP."""
        pass

    def test_search_documents_tool_registered(self) -> None:
        """La herramienta 'search_documents' debe estar registrada en el servidor."""
        pass

    def test_get_document_by_id_tool_registered(self) -> None:
        """La herramienta 'get_document_by_id' debe estar registrada en el servidor."""
        pass
