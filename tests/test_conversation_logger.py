"""
Tests unitarios para ConversationLogger.

Usa base de datos en memoria (:memory:) para evitar I/O y garantizar
que cada test empiece con estado limpio.
"""

from __future__ import annotations

import time

import pytest

from src.agent.conversation_logger import ConversationLogger

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def logger() -> ConversationLogger:
    return ConversationLogger(db_path=":memory:")


def _log(
    lg: ConversationLogger,
    session_id: str = "s1",
    question: str = "¿Qué es?",
    answer: str = "Respuesta",
    sources: list[str] | None = None,
    top_score: float | None = 0.9,
    response_ms: int = 100,
    error: str | None = None,
) -> None:
    lg.log_interaction(
        session_id=session_id,
        question=question,
        answer=answer,
        sources=sources if sources is not None else [],
        top_score=top_score,
        response_ms=response_ms,
        error=error,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_tables_created_on_init(logger: ConversationLogger) -> None:
    """Las tablas conversations y sessions deben existir tras __init__."""
    tables = {row[0] for row in logger.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "conversations" in tables
    assert "sessions" in tables


def test_log_interaction_with_kb_hit(logger: ConversationLogger) -> None:
    """Cuando sources no está vacío, used_kb debe ser 1."""
    _log(logger, sources=["https://bancolombia.com/cuentas"])
    row = logger.conn.execute("SELECT used_kb FROM conversations").fetchone()
    assert row[0] == 1


def test_log_interaction_without_kb_hit(logger: ConversationLogger) -> None:
    """Cuando sources está vacío, used_kb debe ser 0."""
    _log(logger, sources=[])
    row = logger.conn.execute("SELECT used_kb FROM conversations").fetchone()
    assert row[0] == 0


def test_log_creates_session_record(logger: ConversationLogger) -> None:
    """La primera interacción de una sesión debe crear un registro en sessions."""
    _log(logger, session_id="new-session")
    row = logger.conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", ("new-session",)).fetchone()
    assert row is not None


def test_log_updates_existing_session(logger: ConversationLogger) -> None:
    """Dos interacciones de la misma sesión → total_messages = 2."""
    _log(logger, session_id="s-update")
    _log(logger, session_id="s-update", question="¿Otra pregunta?")
    row = logger.conn.execute("SELECT total_messages FROM sessions WHERE session_id = ?", ("s-update",)).fetchone()
    assert row[0] == 2


def test_kb_hit_rate_calculation(logger: ConversationLogger) -> None:
    """Con 2 hits y 1 miss, kb_hit_rate debe ser ~0.666."""
    _log(logger, session_id="s-rate", sources=["https://bancolombia.com/1"])
    _log(logger, session_id="s-rate", sources=["https://bancolombia.com/2"])
    _log(logger, session_id="s-rate", sources=[])
    stats = logger.get_stats()
    assert abs(stats["kb_hit_rate"] - 2 / 3) < 0.001


def test_get_recent_conversations_ordering(logger: ConversationLogger) -> None:
    """Las conversaciones más recientes deben aparecer primero."""
    for i in range(5):
        _log(logger, question=f"Pregunta {i}", response_ms=i * 10)
        time.sleep(0.01)  # asegurar timestamps distintos

    recent = logger.get_recent_conversations(limit=5)
    assert len(recent) == 5
    # La primera debe ser la más reciente (timestamp mayor)
    assert recent[0]["timestamp"] >= recent[-1]["timestamp"]


def test_get_coverage_gaps_returns_unanswered(logger: ConversationLogger) -> None:
    """Las preguntas sin KB (used_kb=0) deben aparecer en coverage gaps."""
    _log(logger, question="¿Precio del dólar?", sources=[])
    _log(logger, question="¿Clima de hoy?", sources=[])
    _log(logger, question="¿Resultados deportivos?", sources=[])

    gaps = logger.get_coverage_gaps()
    gap_questions = {g["question"] for g in gaps}
    assert "¿Precio del dólar?" in gap_questions
    assert "¿Clima de hoy?" in gap_questions
    assert "¿Resultados deportivos?" in gap_questions


def test_logging_error_does_not_raise(logger: ConversationLogger) -> None:
    """Si la conexión está cerrada, log_interaction no debe propagar excepción."""
    logger.close()
    # No debe lanzar ninguna excepción
    logger.log_interaction(
        session_id="s-closed",
        question="¿Test?",
        answer="Resp",
        sources=[],
        top_score=None,
        response_ms=50,
    )


def test_sources_parsed_as_list(logger: ConversationLogger) -> None:
    """get_recent_conversations debe retornar sources como list, no como str."""
    urls = ["https://bancolombia.com/a", "https://bancolombia.com/b"]
    _log(logger, sources=urls)
    recent = logger.get_recent_conversations(limit=1)
    assert len(recent) == 1
    assert isinstance(recent[0]["sources"], list)
    assert recent[0]["sources"] == urls


def test_get_stats_total_counts(logger: ConversationLogger) -> None:
    """get_stats debe contar correctamente total_conversations y total_sessions."""
    _log(logger, session_id="sa")
    _log(logger, session_id="sb")
    _log(logger, session_id="sa")
    stats = logger.get_stats()
    assert stats["total_conversations"] == 3
    assert stats["total_sessions"] == 2


def test_kb_hits_and_misses_tracked_in_sessions(logger: ConversationLogger) -> None:
    """kb_hits y kb_misses en sessions deben reflejar las interacciones."""
    _log(logger, session_id="s-km", sources=["https://bancolombia.com/x"])
    _log(logger, session_id="s-km", sources=[])
    _log(logger, session_id="s-km", sources=["https://bancolombia.com/y"])
    row = logger.conn.execute("SELECT kb_hits, kb_misses FROM sessions WHERE session_id = ?", ("s-km",)).fetchone()
    assert row[0] == 2  # kb_hits
    assert row[1] == 1  # kb_misses
