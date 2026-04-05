"""
MCP Server package: servidor FastMCP que expone herramientas de retrieval.

Proporciona create_server() para construir la instancia configurada del
servidor FastMCP con todas las herramientas registradas, lista para
iniciar la escucha en el host/puerto configurados.
"""

from .server import create_server

__all__ = ["create_server"]
