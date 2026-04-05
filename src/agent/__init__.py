"""
Agent package: agente conversacional RAG basado en OpenAI SDK nativo.

Orquesta el flujo de consulta-respuesta: recibe una pregunta del usuario,
invoca la herramienta MCP search_documents para recuperar contexto y genera
una respuesta fundamentada usando gpt-4o-mini.
"""

from .agent import RAGAgent

__all__ = ["RAGAgent"]
