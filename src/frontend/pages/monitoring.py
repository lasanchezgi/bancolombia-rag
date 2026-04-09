"""
Dashboard de monitoreo interno para el agente RAG de Bancolombia.

Expone métricas de calidad del pipeline RAG, gaps de cobertura,
conversaciones recientes y trazabilidad MCP. Acceso protegido con
contraseña via variable de entorno MONITORING_PASSWORD.

Streamlit detecta automáticamente este archivo en pages/ y lo agrega
como navegación en el sidebar.

Ejecución (junto con app.py):
    uv run streamlit run src/frontend/app.py
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Monitoreo — Bancolombia RAG",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Autenticación
# ──────────────────────────────────────────────────────────────────────────────

MONITORING_PASSWORD = os.getenv("MONITORING_PASSWORD", "bancolombia2026")

if not st.session_state.get("authenticated"):
    st.title("📊 Panel de Monitoreo — Bancolombia RAG")
    st.markdown("Acceso restringido al equipo interno.")
    pwd = st.text_input("🔐 Contraseña de acceso", type="password")
    if pwd == MONITORING_PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Contraseña incorrecta. Intenta de nuevo.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Logger singleton
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_logger():
    """Instancia singleton del ConversationLogger (cacheada entre reruns)."""
    from src.agent.conversation_logger import ConversationLogger  # noqa: PLC0415

    db_path = os.getenv("CONVERSATIONS_DB_PATH", "data/conversations.db")
    return ConversationLogger(db_path=db_path)


logger = get_logger()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Bancolombia RAG")
    st.caption("Dashboard interno de monitoreo")
    st.divider()

    st.subheader("⚙️ Configuración")
    db_path_display = os.getenv("CONVERSATIONS_DB_PATH", "data/conversations.db")
    st.code(db_path_display, language=None)

    st.divider()
    st.subheader("📈 Resumen rápido")

    try:
        _stats = logger.get_stats()
        _total = _stats.get("total_conversations", 0)
        _hit_rate = _stats.get("kb_hit_rate", 0.0)
        st.metric("Conversaciones", _total)

        if _hit_rate >= 0.70:
            _color = "green"
        elif _hit_rate >= 0.50:
            _color = "orange"
        else:
            _color = "red"
        st.markdown(
            f"KB Hit Rate: <span style='color:{_color}; font-weight:bold'>{_hit_rate:.1%}</span>",
            unsafe_allow_html=True,
        )

        _last_convs = logger.get_recent_conversations(limit=1)
        if _last_convs:
            _last_ts = _last_convs[0].get("timestamp", "")
            st.caption(f"Última actividad: {_last_ts[:19].replace('T', ' ')}")
        else:
            st.caption("Sin actividad registrada")
    except Exception:  # noqa: BLE001
        st.warning("No se pudo cargar el resumen.")

    st.divider()
    st.page_link("app.py", label="← Ir al Chat")

# ──────────────────────────────────────────────────────────────────────────────
# Sección 1 — Header y KPIs
# ──────────────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("📊 Panel de Monitoreo — Bancolombia RAG Assistant")
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Actualizado: {now_str}")
with col_btn:
    st.write("")
    if st.button("🔄 Actualizar datos", use_container_width=True):
        st.rerun()

st.divider()

try:
    stats = logger.get_stats()
    total_conversations = stats.get("total_conversations", 0)
    kb_hit_rate = stats.get("kb_hit_rate", 0.0)
    avg_response_ms = stats.get("avg_response_ms", 0.0)
    no_kb_count = total_conversations - round(total_conversations * kb_hit_rate)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("💬 Conversaciones", total_conversations)
    with kpi2:
        st.metric("🎯 KB Hit Rate", f"{kb_hit_rate:.1%}")
    with kpi3:
        avg_s = avg_response_ms / 1000 if avg_response_ms else 0.0
        st.metric("⚡ Tiempo promedio", f"{avg_s:.1f}s")
    with kpi4:
        st.metric("⚠️ Sin respuesta KB", no_kb_count)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Error cargando métricas globales: {exc}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 2 — Calidad del RAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────

st.header("🔬 Calidad del RAG Pipeline")

try:
    metrics = logger.get_rag_performance_metrics()

    col_scores, col_times = st.columns(2)

    with col_scores:
        st.subheader("🔍 Scores de Retrieval")

        avg_chromadb = metrics.get("avg_chromadb_score", 0.0)
        avg_rerank = metrics.get("avg_rerank_score", 0.0)
        reranking_improvement = metrics.get("reranking_improvement", 0.0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Score ChromaDB", f"{avg_chromadb:.3f}")
        with m2:
            st.metric("Score Reranking", f"{avg_rerank:.3f}")
        with m3:
            sign = "+" if reranking_improvement >= 0 else ""
            delta_str = f"{sign}{reranking_improvement:.3f}"
            st.metric("Mejora Reranking", f"{reranking_improvement:.3f}", delta=delta_str)

        if avg_chromadb > 0 or avg_rerank > 0:
            scores_df = pd.DataFrame(
                {"Score": [avg_chromadb, avg_rerank]},
                index=["ChromaDB", "Reranking"],
            )
            st.bar_chart(scores_df, height=200)
        else:
            st.info("Sin datos de scores aún.")

    with col_times:
        st.subheader("⏱️ Performance del Pipeline")

        avg_retrieval = metrics.get("avg_retrieval_ms", 0.0)
        avg_reranking = metrics.get("avg_reranking_ms", 0.0)
        avg_total = metrics.get("avg_total_ms", 0.0)
        total_calls = metrics.get("total_mcp_calls", 0)
        calls_rerank = metrics.get("calls_with_reranking", 0)
        calls_no_rerank = metrics.get("calls_without_reranking", 0)

        timing_df = pd.DataFrame(
            {
                "Etapa": ["Retrieval (ChromaDB)", "Reranking (Cross-Encoder)", "Total MCP"],
                "Tiempo promedio (ms)": [
                    round(avg_retrieval),
                    round(avg_reranking),
                    round(avg_total),
                ],
            }
        )
        st.dataframe(timing_df, hide_index=True, use_container_width=True)

        if avg_total > 0:
            retrieval_pct = min(avg_retrieval / avg_total, 1.0)
            reranking_pct = min(avg_reranking / avg_total, 1.0)
            st.caption("Proporción Retrieval")
            st.progress(retrieval_pct)
            st.caption("Proporción Reranking")
            st.progress(reranking_pct)

        st.caption(
            f"Total llamadas MCP: **{total_calls}** "
            f"(con reranking: {calls_rerank}, sin reranking: {calls_no_rerank})"
        )

except Exception as exc:  # noqa: BLE001
    st.warning(f"Error cargando métricas RAG: {exc}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 3 — Gaps de Cobertura
# ──────────────────────────────────────────────────────────────────────────────

st.header("⚠️ Gaps de Cobertura Detectados")
st.caption("Preguntas que el agente no pudo responder con la base de conocimiento")

tab_no_kb, tab_low_score = st.tabs(["🔴 Sin respuesta de KB", "📉 Retrieval de baja calidad"])

with tab_no_kb:
    try:
        gaps = logger.get_coverage_gaps(limit=20)
        if not gaps:
            st.success("✅ No se detectaron gaps de cobertura.")
        else:
            gaps_df = pd.DataFrame(
                [
                    {
                        "Pregunta": g.get("question", ""),
                        "Veces preguntada": g.get("count", 0),
                        "Última vez": (g.get("last_asked", "") or "")[:19].replace("T", " "),
                    }
                    for g in gaps
                ]
            )
            st.dataframe(gaps_df, hide_index=True, use_container_width=True)

            csv_data = gaps_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Exportar CSV",
                data=csv_data,
                file_name="gaps_cobertura.csv",
                mime="text/csv",
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Error cargando gaps: {exc}")

with tab_low_score:
    try:
        low_score_data = metrics.get("low_score_queries", []) if "metrics" in dir() else []
        if not low_score_data:
            # Fallback: filtrar desde traces
            all_traces = logger.get_mcp_traces(tool_name="search_knowledge_base", limit=200)
            low_score_data = [
                {
                    "query": t.get("query_sent", ""),
                    "top_chromadb_score": t.get("top_chromadb_score"),
                    "timestamp": t.get("timestamp", ""),
                }
                for t in all_traces
                if t.get("top_chromadb_score") is not None and t.get("top_chromadb_score", 1.0) < 0.5
            ]

        if not low_score_data:
            st.success("✅ Todos los retrievals tienen score aceptable.")
        else:
            low_df = pd.DataFrame(
                [
                    {
                        "Query enviado al MCP": item.get("query", ""),
                        "Score ChromaDB": round(item.get("top_chromadb_score", 0.0) or 0.0, 4),
                        "Timestamp": (item.get("timestamp", "") or "")[:19].replace("T", " "),
                    }
                    for item in low_score_data
                ]
            )
            st.dataframe(low_df, hide_index=True, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Error cargando retrieval de baja calidad: {exc}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 4 — Conversaciones Recientes
# ──────────────────────────────────────────────────────────────────────────────

st.header("💬 Conversaciones Recientes")

filter_col, slider_col = st.columns([2, 1])
with filter_col:
    conv_filter = st.selectbox(
        "Filtrar por tipo",
        ["Todas", "Con KB", "Sin KB", "Con error"],
        key="conv_filter",
    )
with slider_col:
    conv_limit = st.slider("Últimas N conversaciones", min_value=10, max_value=100, value=20, step=10)

try:
    conversations = logger.get_recent_conversations(limit=conv_limit)

    # Aplicar filtro
    if conv_filter == "Con KB":
        conversations = [c for c in conversations if c.get("used_kb") == 1]
    elif conv_filter == "Sin KB":
        conversations = [c for c in conversations if c.get("used_kb") == 0]
    elif conv_filter == "Con error":
        conversations = [c for c in conversations if c.get("error")]

    if not conversations:
        st.info("📭 No hay conversaciones registradas aún. Interactúa con el chat para ver datos aquí.")
    else:
        for conv in conversations:
            ts = (conv.get("timestamp") or "")[:19].replace("T", " ")
            question = conv.get("question") or ""
            answer = conv.get("answer") or ""
            sources = conv.get("sources") or []
            used_kb = conv.get("used_kb", 0)
            top_score = conv.get("top_score")
            response_ms = conv.get("response_ms") or 0
            error = conv.get("error")
            session_id = conv.get("session_id", "")

            kb_icon = "✅" if used_kb else "❌"
            score_str = f"{top_score:.3f}" if top_score is not None else "N/A"
            q_preview = question[:60] + ("..." if len(question) > 60 else "")
            expander_label = f"{ts}  |  {q_preview}  |  KB: {kb_icon}  |  Score: {score_str}  |  {response_ms}ms"

            with st.expander(expander_label):
                st.markdown("**Pregunta completa:**")
                st.write(question)

                st.markdown("**Respuesta:**")
                st.write(answer)

                if sources:
                    st.markdown("**Fuentes utilizadas:**")
                    for url in sources:
                        st.markdown(f"- [{url}]({url})")
                else:
                    st.caption("Sin fuentes (KB no utilizada)")

                if error:
                    st.error(f"Error registrado: {error}")

                # MCP traces de esta sesión
                if session_id:
                    mcp_traces = logger.get_mcp_traces(session_id=session_id, limit=5)
                    if mcp_traces:
                        st.markdown("**Llamadas MCP en esta sesión:**")
                        for trace in mcp_traces:
                            tool = trace.get("tool_name", "")
                            query = trace.get("query_sent") or ""
                            chroma = trace.get("top_chromadb_score")
                            rerank = trace.get("top_rerank_score")
                            ret_ms = trace.get("retrieval_ms") or 0
                            rer_ms = trace.get("reranking_ms") or 0
                            chroma_str = f"{chroma:.3f}" if chroma is not None else "N/A"
                            rerank_str = f"{rerank:.3f}" if rerank is not None else "N/A"
                            st.caption(
                                f"🔌 `{tool}` | query: `{query[:50]}` | "
                                f"chromadb: {chroma_str} | rerank: {rerank_str} | "
                                f"retrieval: {ret_ms}ms | reranking: {rer_ms}ms"
                            )

except Exception as exc:  # noqa: BLE001
    st.warning(f"Error cargando conversaciones: {exc}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 5 — Trazabilidad MCP
# ──────────────────────────────────────────────────────────────────────────────

st.header("🔌 Trazabilidad del Servidor MCP")

try:
    col_calls, col_freq = st.columns(2)

    with col_calls:
        st.subheader("Llamadas por tool")
        search_traces = logger.get_mcp_traces(tool_name="search_knowledge_base", limit=10000)
        article_traces = logger.get_mcp_traces(tool_name="get_article_by_url", limit=10000)
        categories_traces = logger.get_mcp_traces(tool_name="list_categories", limit=10000)

        st.metric("search_knowledge_base", len(search_traces))
        st.metric("get_article_by_url", len(article_traces))
        st.metric("list_categories", len(categories_traces))

    with col_freq:
        st.subheader("Queries más frecuentes al MCP")
        freq_queries = metrics.get("most_frequent_queries", []) if "metrics" in dir() else []
        if not freq_queries:
            st.info("Sin queries registrados aún.")
        else:
            freq_df = pd.DataFrame(
                [
                    {
                        "Query": q.get("query", ""),
                        "Frecuencia": q.get("count", 0),
                    }
                    for q in freq_queries
                ]
            )
            st.dataframe(freq_df, hide_index=True, use_container_width=True)

except Exception as exc:  # noqa: BLE001
    st.warning(f"Error cargando trazabilidad MCP: {exc}")
