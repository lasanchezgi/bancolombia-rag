"""
Agente conversacional RAG usando el SDK nativo de OpenAI y MCP via stdio.

Implementa el loop agéntico completo: se conecta al servidor MCP de
Bancolombia, recupera contexto relevante a través de las herramientas MCP,
y genera respuestas con gpt-4o-mini usando tool use nativo.

Sin dependencias en LangChain ni frameworks de agentes externos.
La sesión MCP se mantiene abierta en un hilo dedicado para evitar
el costo de re-inicialización en cada llamada.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from .memory import LongTermMemory, MidTermMemory, ShortTermMemory
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_MID_TERM_THRESHOLD = 15


class RAGAgent:
    """Agente conversacional RAG con conexión MCP a la base de conocimiento Bancolombia.

    Attributes:
        client: Cliente OpenAI para generación de respuestas.
        mcp_server_script: Ruta al script del servidor MCP.
        short_term: Memoria de corto plazo (ventana deslizante).
        mid_term: Memoria de mediano plazo (resumen).
        long_term: Memoria de largo plazo (perfil persistido).
    """

    TOOLS: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": (
                    "Busca información sobre productos, servicios y contenido de Bancolombia "
                    "en la base de conocimiento. Usar cuando el usuario pregunte sobre cuentas, "
                    "tarjetas, créditos, beneficios, giros o cualquier producto bancario."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Pregunta o consulta en lenguaje natural",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número de resultados (default: 5)",
                            "default": 5,
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "Filtrar por categoría: cuentas, tarjetas-de-credito, "
                                "tarjetas-debito, creditos, beneficios, giros, general"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_article_by_url",
                "description": (
                    "Recupera contenido completo de una página de Bancolombia por URL. "
                    "Usar cuando se necesiten más detalles de una página específica."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL completa de la página de Bancolombia",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_categories",
                "description": (
                    "Lista categorías disponibles en la base de conocimiento. "
                    "Usar cuando el usuario pregunte qué temas cubre el asistente."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]

    def __init__(
        self,
        openai_api_key: str,
        mcp_server_script: str = "src/mcp_server/server.py",
    ) -> None:
        self.client = OpenAI(api_key=openai_api_key)
        self.mcp_server_script = mcp_server_script
        self.short_term = ShortTermMemory(max_messages=20)
        self.mid_term = MidTermMemory()
        self.long_term = LongTermMemory()
        self.long_term.load()
        self._mcp_session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        # Dedicated event loop in a background daemon thread.
        # Keeps the MCP stdio_client context alive across all ask() calls.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    # ──────────────────────────────────────────────────────────────────────
    # MCP session management
    # ──────────────────────────────────────────────────────────────────────

    async def _get_mcp_session(self) -> ClientSession:
        """Inicializa conexión MCP via stdio (lazy, singleton por instancia).

        Usa sys.executable para asegurar que se use el Python del entorno
        virtual correcto (funciona con uv y virtualenvs).
        """
        if self._mcp_session is not None:
            return self._mcp_session

        self._exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.mcp_server_script],
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self._mcp_session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._mcp_session.initialize()
        logger.info("Sesión MCP inicializada con %s", self.mcp_server_script)
        return self._mcp_session

    async def _call_mcp_tool(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Llama una tool del servidor MCP y retorna el resultado parseado.

        Args:
            tool_name: Nombre de la tool MCP a invocar.
            tool_args: Argumentos de la tool.

        Returns:
            Resultado de la tool como dict. Si hay error, retorna {"error": str}.
        """
        try:
            session = await self._get_mcp_session()
            result = await session.call_tool(tool_name, tool_args)
            first = result.content[0]
            text: str = first.text if hasattr(first, "text") else str(first)  # type: ignore[union-attr]
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error llamando tool MCP '%s': %s", tool_name, exc)
            return {"error": str(exc)}

    # ──────────────────────────────────────────────────────────────────────
    # Agentic loop
    # ──────────────────────────────────────────────────────────────────────

    async def _run_agentic_loop(self, user_message: str) -> str:
        """Loop agéntico completo: recibe mensaje, invoca tools, retorna respuesta.

        1. Agrega user_message a ShortTermMemory.
        2. Si hay > 15 mensajes, actualiza MidTermMemory con resumen.
        3. Construye lista de mensajes para OpenAI (con contextos de memoria).
        4. Loop OpenAI: tool_calls → _call_mcp_tool → continuar; stop → retornar.

        Args:
            user_message: Mensaje del usuario.

        Returns:
            Respuesta final del asistente como string.
        """
        self.short_term.add_message("user", user_message)

        if len(self.short_term.get_messages()) > _MID_TERM_THRESHOLD:
            self.mid_term.update_summary(self.short_term.get_messages(), self.client)

        messages: list[dict[str, Any]] = list(self.short_term.get_messages())

        # Inyectar resumen de mid-term como contexto adicional si existe
        summary = self.mid_term.get_summary()
        if summary:
            messages = (
                [messages[0]]
                + [{"role": "system", "content": f"Resumen de la conversación hasta ahora: {summary}"}]
                + messages[1:]
            )

        while True:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,  # type: ignore[arg-type]
                tools=self.TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
            )
            choice = response.choices[0]
            assistant_message = choice.message

            if choice.finish_reason == "tool_calls" and assistant_message.tool_calls:
                # Agregar mensaje del asistente con tool_calls al historial temporal
                messages.append(assistant_message.model_dump(exclude_unset=False))

                for tool_call in assistant_message.tool_calls:
                    tool_name: str = tool_call.function.name  # type: ignore[union-attr]
                    try:
                        tool_args = json.loads(tool_call.function.arguments)  # type: ignore[union-attr]
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_result = await self._call_mcp_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )

            elif choice.finish_reason == "stop":
                final_text = assistant_message.content or ""
                self.short_term.add_message("assistant", final_text)
                self.long_term.update_from_conversation(self.short_term.get_messages())
                return final_text

            else:
                # Finish reason inesperado (content_filter, length, etc.)
                return assistant_message.content or "No pude generar una respuesta."

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def ask(self, user_message: str) -> str:
        """Procesa una pregunta y retorna la respuesta del agente (sincrónico).

        Inicializa el system message en la primera llamada.
        Delega la ejecución al loop dedicado del hilo de fondo.

        Args:
            user_message: Pregunta del usuario.

        Returns:
            Respuesta generada por el agente.
        """
        if not self.short_term.get_messages():
            self.short_term.add_message(
                "system",
                SYSTEM_PROMPT.format(long_term_context=self.long_term.get_context()),
            )

        future = asyncio.run_coroutine_threadsafe(
            self._run_agentic_loop(user_message),
            self._loop,
        )
        return future.result(timeout=120)

    def reset_conversation(self) -> None:
        """Limpia ShortTermMemory y MidTermMemory preservando el system message."""
        self.short_term.clear()
        self.mid_term.summary = None

    def get_history(self) -> list[dict[str, str]]:
        """Retorna historial de mensajes para mostrar en el frontend."""
        return self.short_term.get_messages()
