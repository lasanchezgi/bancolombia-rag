"""
Memoria conversacional de corto plazo para el agente RAG.

Mantiene una ventana deslizante de los últimos N turnos de conversación
como lista de dicts en el formato de mensajes de OpenAI (``role`` + ``content``).
La memoria es in-memory: no persiste entre reinicios del proceso.
"""

from __future__ import annotations


class ConversationMemory:
    """Gestiona el historial reciente de la conversación del agente.

    Implementa una ventana de tamaño fijo: cuando se supera ``max_turns``,
    los turnos más antiguos se eliminan automáticamente para no exceder
    el contexto del modelo.

    Attributes:
        max_turns: Número máximo de pares (user, assistant) a conservar.
    """

    def __init__(self, max_turns: int = 10) -> None:
        """Inicializa la memoria con una ventana vacía.

        Args:
            max_turns: Máximo de turnos a conservar en el historial.
                       Un turno = un mensaje de usuario + un mensaje de asistente.
        """
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """Añade un mensaje al historial de conversación.

        Si el historial supera ``max_turns * 2`` mensajes (contando ambos roles),
        se elimina el par más antiguo para mantener la ventana.

        Args:
            role: Rol del mensaje; debe ser ``"user"`` o ``"assistant"``.
            content: Contenido del mensaje.
        """
        raise NotImplementedError

    def get_history(self) -> list[dict[str, str]]:
        """Devuelve el historial de conversación actual.

        Returns:
            Lista de dicts con las claves ``role`` y ``content``,
            en orden cronológico de más antiguo a más reciente.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Vacía el historial de conversación."""
        raise NotImplementedError
