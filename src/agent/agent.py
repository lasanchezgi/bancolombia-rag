"""
Agente conversacional RAG usando el SDK nativo de OpenAI.

Implementa el loop agéntico completo: recupera contexto relevante a través
de las herramientas MCP, construye el prompt con ese contexto e historial
de conversación, y genera la respuesta final con gpt-4o-mini.

Sin dependencias en LangChain ni frameworks de agentes externos.
"""

from __future__ import annotations

from openai import OpenAI

from .memory import ConversationMemory


class RAGAgent:
    """Agente conversacional que fundamenta sus respuestas en documentos recuperados.

    Attributes:
        model: Identificador del modelo de chat de OpenAI.
        client: Instancia del cliente OpenAI.
        memory: Historial de conversación de corto plazo.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Inicializa el agente con su modelo, cliente y memoria.

        Args:
            model: Modelo de OpenAI a usar para la generación de respuestas.
        """
        self.model = model
        self.client = OpenAI()
        self.memory = ConversationMemory()

    def ask(self, question: str) -> str:
        """Procesa una pregunta y devuelve una respuesta fundamentada.

        Flujo interno:
        1. Recupera documentos relevantes llamando a search_documents via MCP.
        2. Construye el prompt con contexto + historial de conversación.
        3. Llama a la API de chat de OpenAI.
        4. Guarda el turno en memoria y devuelve la respuesta.

        Args:
            question: Pregunta en lenguaje natural del usuario.

        Returns:
            Respuesta generada por el modelo, fundamentada en los documentos
            recuperados del vector store.
        """
        raise NotImplementedError

    def _build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        """Construye la lista de mensajes para la llamada a la API de chat.

        Args:
            question: Pregunta actual del usuario.
            context: Texto de los documentos recuperados por el retriever.

        Returns:
            Lista de dicts con ``role`` y ``content`` en el formato de OpenAI.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reinicia la memoria conversacional del agente."""
        self.memory.clear()
