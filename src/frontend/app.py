"""
Aplicación Streamlit: interfaz de chat para el agente RAG de Bancolombia.

Renderiza un chat interactivo que envía mensajes al RAGAgent y muestra
las respuestas. El estado de sesión de Streamlit persiste el historial
de mensajes y la instancia del agente entre reruns.

Ejecución:
    uv run streamlit run src/frontend/app.py
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Asistente Bancolombia",
    page_icon="🏦",
    layout="centered",
)

_CATEGORIES = [
    "Cuentas de ahorro y corriente",
    "Tarjetas de crédito",
    "Tarjetas débito",
    "Créditos de consumo y vivienda",
    "Beneficios y promociones",
    "Giros internacionales",
]


def _get_agent():
    """Importa y retorna una instancia de RAGAgent (import diferido para evitar errores en carga)."""
    from src.agent.agent import RAGAgent  # noqa: PLC0415

    return RAGAgent(openai_api_key=os.environ["OPENAI_API_KEY"])


def init_session() -> None:
    """Inicializa el estado de sesión de Streamlit en el primer cargue.

    Crea la instancia del RAGAgent y la lista de mensajes del chat
    si aún no existen en st.session_state. Idempotente.
    """
    if "agent" not in st.session_state:
        st.session_state.agent = _get_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def render_sidebar() -> None:
    """Renderiza el panel lateral con información y controles."""
    with st.sidebar:
        st.header("ℹ️ Acerca de este asistente")
        st.write(
            "Asistente virtual especializado en productos y servicios "
            "de Bancolombia para personas naturales. "
            "Las respuestas se basan en la información publicada en el sitio web oficial."
        )
        st.divider()

        st.subheader("📊 Estado")
        n_messages = len(st.session_state.get("messages", []))
        st.write(f"Mensajes en conversación: **{n_messages}**")

        st.divider()
        st.subheader("🔍 Temas disponibles")
        for cat in _CATEGORIES:
            st.write(f"• {cat}")

        st.divider()
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.agent.reset_conversation()
            st.session_state.messages = []
            st.rerun()


def render_chat() -> None:
    """Renderiza el historial de mensajes y procesa el input del usuario."""
    st.title("🏦 Asistente Virtual Bancolombia")
    st.caption("Consulta sobre productos y servicios · Basado en información oficial")

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Procesar nuevo input
    if prompt := st.chat_input("¿En qué te puedo ayudar?"):
        # Mostrar mensaje del usuario inmediatamente
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtener respuesta del agente
        with st.chat_message("assistant"):
            with st.spinner("Consultando base de conocimiento..."):
                response = st.session_state.agent.ask(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────────────────────

init_session()
render_sidebar()
render_chat()
