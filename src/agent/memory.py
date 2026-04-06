"""
Módulos de memoria para el agente RAG de Bancolombia.

Implementa tres capas de memoria:
- ShortTermMemory: historial activo con ventana deslizante (max 20 mensajes)
- MidTermMemory: resumen generado por LLM cuando el historial crece
- LongTermMemory: perfil del usuario persistido en JSON entre sesiones
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BANCOLOMBIA_KEYWORDS: dict[str, list[str]] = {
    "cuentas": ["cuenta", "ahorro", "corriente", "nómina", "nomina"],
    "tarjetas-de-credito": ["tarjeta de crédito", "tarjeta credito", "visa", "mastercard", "amex", "american express"],
    "tarjetas-debito": ["tarjeta débito", "tarjeta debito", "débito", "debito"],
    "creditos": ["crédito", "credito", "préstamo", "prestamo", "hipoteca", "vivienda"],
    "beneficios": ["beneficio", "promoción", "promocion", "descuento"],
    "giros": ["giro", "transferencia internacional", "remesa"],
}


class ShortTermMemory:
    """Memoria de corto plazo: historial de la conversación activa.

    Implementa ventana deslizante para no exceder el context window.
    El mensaje de sistema (índice 0) NUNCA se elimina.
    """

    def __init__(self, max_messages: int = 20) -> None:
        self._messages: list[dict[str, str]] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str) -> None:
        """Agrega un mensaje al historial y aplica la ventana deslizante."""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_messages:
            # Preservar system (index 0) + los (max_messages - 1) más recientes
            self._messages = [self._messages[0]] + self._messages[-(self.max_messages - 1):]

    def get_messages(self) -> list[dict[str, str]]:
        """Retorna historial completo para pasar a OpenAI."""
        return list(self._messages)

    def clear(self) -> None:
        """Limpia historial manteniendo solo el system message."""
        if self._messages:
            self._messages = [self._messages[0]]
        else:
            self._messages = []


class MidTermMemory:
    """Memoria de mediano plazo: resumen de la conversación actual.

    Se genera cuando ShortTermMemory supera los 15 mensajes para mantener
    contexto de temas discutidos sin saturar el context window.
    """

    def __init__(self) -> None:
        self.summary: str | None = None

    def update_summary(self, messages: list[dict[str, Any]], openai_client: Any) -> None:
        """Genera resumen cuando hay más de 15 mensajes en short term.

        Args:
            messages: Lista de mensajes del historial de conversación.
            openai_client: Instancia del cliente OpenAI para generar el resumen.
        """
        if len(messages) <= 15:
            return
        try:
            conversation_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in messages
                if m["role"] in ("user", "assistant")
            )
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Resume en 3-5 oraciones los temas discutidos "
                            "en esta conversación sobre productos Bancolombia:\n\n"
                            f"{conversation_text}"
                        ),
                    }
                ],
                max_tokens=200,
            )
            self.summary = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            logger.error("Error generando resumen mid-term: %s", exc)

    def get_summary(self) -> str | None:
        """Retorna el resumen actual o None si no hay."""
        return self.summary


class LongTermMemory:
    """Memoria de largo plazo: preferencias y contexto del usuario.

    Persiste entre sesiones en un archivo JSON local.
    Guarda temas consultados, productos de interés y conteo de sesiones.
    """

    def __init__(self, storage_path: Path = Path(".memory/user_profile.json")) -> None:
        self.storage_path = storage_path
        self.data: dict[str, Any] = {
            "topics_consulted": [],
            "products_of_interest": [],
            "session_count": 0,
            "last_session": "",
        }

    def load(self) -> None:
        """Carga perfil desde storage_path si existe."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo cargar long-term memory: %s", exc)

    def save(self) -> None:
        """Persiste perfil en storage_path (crea directorio si no existe)."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error guardando long-term memory: %s", exc)

    def update_from_conversation(self, messages: list[dict[str, Any]]) -> None:
        """Extrae temas y productos mencionados, actualiza contadores y guarda."""
        combined_text = " ".join(
            m["content"].lower()
            for m in messages
            if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
        )

        for topic, keywords in _BANCOLOMBIA_KEYWORDS.items():
            if any(kw in combined_text for kw in keywords):
                if topic not in self.data["topics_consulted"]:
                    self.data["topics_consulted"].append(topic)
                if topic not in self.data["products_of_interest"]:
                    self.data["products_of_interest"].append(topic)

        self.data["session_count"] = self.data.get("session_count", 0) + 1
        self.data["last_session"] = datetime.now().isoformat()
        self.save()

    def get_context(self) -> str:
        """Retorna string con contexto para incluir en system prompt."""
        topics = self.data.get("topics_consulted", [])
        if not topics:
            return ""
        topics_str = ", ".join(topics)
        return f"El usuario ha consultado antes sobre: {topics_str}."
