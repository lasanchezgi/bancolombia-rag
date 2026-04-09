"""
Tests unitarios para trazabilidad del pipeline RAG (mcp_calls).

Usa base de datos en memoria (:memory:) para evitar I/O y garantizar
que cada test empiece con estado limpio.
"""

from __future__ import annotations

import pytest

from src.agent.conversation_logger import ConversationLogger

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def lg() -> ConversationLogger:
    return ConversationLogger(db_path=":memory:")


def _log_mcp(
    lg: ConversationLogger,
    session_id: str = "s1",
    tool_name: str = "search_knowledge_base",
    query_sent: str | None = "¿Qué es?",
    top_chromadb_score: float | None = 0.8,
    top_rerank_score: float | None = None,
    urls_returned: list[str] | None = None,
    titles_returned: list[str] | None = None,
    retrieval_ms: int = 50,
    reranking_ms: int = 0,
    total_ms: int = 50,
    chunks_retrieved: int = 5,
    chunks_after_rerank: int = 5,
) -> int:
    return lg.log_mcp_call(
        session_id=session_id,
        tool_name=tool_name,
        query_sent=query_sent,
        top_chromadb_score=top_chromadb_score,
        top_rerank_score=top_rerank_score,
        urls_returned=urls_returned or [],
        titles_returned=titles_returned or [],
        retrieval_ms=retrieval_ms,
        reranking_ms=reranking_ms,
        total_ms=total_ms,
        chunks_retrieved=chunks_retrieved,
        chunks_after_rerank=chunks_after_rerank,
    )


def _log_interaction(
    lg: ConversationLogger,
    session_id: str = "s1",
    question: str = "¿Qué es?",
    answer: str = "Respuesta",
    sources: list[str] | None = None,
    top_score: float | None = 0.9,
    response_ms: int = 100,
) -> int:
    return lg.log_interaction(
        session_id=session_id,
        question=question,
        answer=answer,
        sources=sources if sources is not None else [],
        top_score=top_score,
        response_ms=response_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_mcp_calls_table_created(lg: ConversationLogger) -> None:
    """Tabla mcp_calls debe existir tras init."""
    tables = {row[0] for row in lg.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "mcp_calls" in tables


def test_log_mcp_call_search_kb(lg: ConversationLogger) -> None:
    """log_mcp_call con tool_name=search_knowledge_base debe crear registro en DB."""
    _log_mcp(lg, tool_name="search_knowledge_base")
    row = lg.conn.execute("SELECT tool_name FROM mcp_calls").fetchone()
    assert row is not None
    assert row["tool_name"] == "search_knowledge_base"


def test_log_mcp_call_serializes_urls_as_json(lg: ConversationLogger) -> None:
    """urls_returned debe guardarse como JSON string y get_mcp_traces retorna list."""
    urls = ["https://bancolombia.com/1", "https://bancolombia.com/2"]
    _log_mcp(lg, urls_returned=urls)

    # En DB debe estar como string JSON
    raw = lg.conn.execute("SELECT urls_returned FROM mcp_calls").fetchone()["urls_returned"]
    assert isinstance(raw, str)
    assert raw.startswith("[")

    # get_mcp_traces debe parsear a list Python
    traces = lg.get_mcp_traces()
    assert len(traces) == 1
    assert isinstance(traces[0]["urls_returned"], list)
    assert traces[0]["urls_returned"] == urls


def test_log_mcp_call_returns_id(lg: ConversationLogger) -> None:
    """log_mcp_call debe retornar int > 0."""
    mcp_id = _log_mcp(lg)
    assert isinstance(mcp_id, int)
    assert mcp_id > 0


def test_log_mcp_call_returns_minus_one_on_error(lg: ConversationLogger) -> None:
    """Con conexión cerrada, log_mcp_call debe retornar -1 sin propagar excepción."""
    lg.close()
    result = lg.log_mcp_call(session_id="s1", tool_name="search_knowledge_base")
    assert result == -1


def test_get_mcp_traces_filter_by_tool(lg: ConversationLogger) -> None:
    """Filtro por tool_name debe retornar solo las llamadas de esa tool."""
    _log_mcp(lg, tool_name="search_knowledge_base")
    _log_mcp(lg, tool_name="search_knowledge_base")
    _log_mcp(lg, tool_name="get_article_by_url")

    traces = lg.get_mcp_traces(tool_name="get_article_by_url")
    assert len(traces) == 1
    assert traces[0]["tool_name"] == "get_article_by_url"


def test_get_mcp_traces_filter_by_session(lg: ConversationLogger) -> None:
    """Filtro por session_id debe retornar solo las llamadas de esa sesión."""
    _log_mcp(lg, session_id="session-A")
    _log_mcp(lg, session_id="session-A")
    _log_mcp(lg, session_id="session-B")

    traces_a = lg.get_mcp_traces(session_id="session-A")
    traces_b = lg.get_mcp_traces(session_id="session-B")

    assert len(traces_a) == 2
    assert len(traces_b) == 1
    assert all(t["session_id"] == "session-A" for t in traces_a)


def test_rag_performance_metrics_avg_scores(lg: ConversationLogger) -> None:
    """avg_chromadb_score debe ser el promedio correcto de los scores registrados."""
    _log_mcp(lg, top_chromadb_score=0.6)
    _log_mcp(lg, top_chromadb_score=0.8)
    _log_mcp(lg, top_chromadb_score=1.0)

    metrics = lg.get_rag_performance_metrics()
    expected_avg = (0.6 + 0.8 + 1.0) / 3
    assert abs(metrics["avg_chromadb_score"] - expected_avg) < 0.001


def test_rag_performance_reranking_improvement(lg: ConversationLogger) -> None:
    """reranking_improvement debe reflejar la diferencia promedio rerank - chromadb."""
    lg.log_mcp_call(
        session_id="s1",
        tool_name="search_knowledge_base",
        top_chromadb_score=0.7,
        top_rerank_score=0.9,
        use_reranking=True,
    )

    metrics = lg.get_rag_performance_metrics()
    assert abs(metrics["reranking_improvement"] - 0.2) < 0.001


def test_most_frequent_queries(lg: ConversationLogger) -> None:
    """La query más repetida debe aparecer primera en most_frequent_queries."""
    for _ in range(3):
        _log_mcp(lg, query_sent="tarjetas de crédito")
    _log_mcp(lg, query_sent="beneficios AFC")

    metrics = lg.get_rag_performance_metrics()
    assert len(metrics["most_frequent_queries"]) >= 1
    assert metrics["most_frequent_queries"][0]["query"] == "tarjetas de crédito"
    assert metrics["most_frequent_queries"][0]["count"] == 3


def test_low_score_queries_threshold(lg: ConversationLogger) -> None:
    """low_score_queries debe incluir solo queries con top_chromadb_score < 0.5."""
    _log_mcp(lg, query_sent="dólar hoy", top_chromadb_score=0.3)
    _log_mcp(lg, query_sent="clima bogotá", top_chromadb_score=0.3)
    _log_mcp(lg, query_sent="tarjetas débito", top_chromadb_score=0.8)

    metrics = lg.get_rag_performance_metrics()
    low = metrics["low_score_queries"]
    low_queries = {q["query"] for q in low}
    assert "dólar hoy" in low_queries
    assert "clima bogotá" in low_queries
    assert "tarjetas débito" not in low_queries


def test_full_pipeline_trace(lg: ConversationLogger) -> None:
    """Simular sesión completa: log_interaction + 2x log_mcp_call."""
    # Registrar dos llamadas MCP primero (sin conversation_id)
    mcp_id1 = _log_mcp(lg, session_id="full-session", top_chromadb_score=0.75, top_rerank_score=0.88)
    mcp_id2 = _log_mcp(lg, session_id="full-session", top_chromadb_score=0.65, top_rerank_score=0.80)

    assert mcp_id1 > 0
    assert mcp_id2 > 0

    # Registrar interacción y vincular
    conv_id = _log_interaction(
        lg,
        session_id="full-session",
        sources=["https://bancolombia.com/tarjetas"],
    )
    assert conv_id > 0

    lg._link_mcp_calls_to_conversation([mcp_id1, mcp_id2], conv_id)  # noqa: SLF001

    # Verificar FK en DB
    linked = lg.conn.execute("SELECT id FROM mcp_calls WHERE conversation_id = ?", (conv_id,)).fetchall()
    assert len(linked) == 2

    # get_stats debe reflejar la conversación
    stats = lg.get_stats()
    assert stats["total_conversations"] == 1
    assert stats["kb_hit_rate"] == 1.0

    # get_rag_performance_metrics debe tener datos
    metrics = lg.get_rag_performance_metrics()
    assert metrics["total_mcp_calls"] == 2
    assert metrics["avg_chromadb_score"] > 0
    assert metrics["avg_rerank_score"] > 0
