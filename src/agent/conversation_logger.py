"""
Logger de conversaciones del agente RAG.

Persiste cada interacción en SQLite para análisis posterior.
Permite auditar preguntas, identificar gaps en la KB, medir rendimiento
y detectar errores. Cero costo, cero servicios externos.

En producción multi-instancia migrar a PostgreSQL cambiando solo este adaptador.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logger de conversaciones del agente RAG.

    Persiste cada interacción en SQLite para análisis posterior.

    Permite:
    - Auditar qué preguntas se hacen al agente
    - Identificar gaps en la base de conocimiento
    - Medir rendimiento del sistema (tiempos, scores)
    - Detectar errores y preguntas fuera de scope

    Storage: SQLite (data/conversations.db)
    No requiere servicios externos. En producción multi-instancia
    migrar a PostgreSQL/RDS cambiando solo este adaptador.
    """

    def __init__(self, db_path: str = "data/conversations.db") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("ConversationLogger inicializado: %s", db_path)

    def _create_tables(self) -> None:
        """Crea las tablas y los índices si no existen."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                question        TEXT NOT NULL,
                answer          TEXT NOT NULL,
                sources         TEXT,
                top_score       REAL,
                used_kb         INTEGER,
                response_ms     INTEGER,
                error           TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                started_at      TEXT NOT NULL,
                last_active     TEXT NOT NULL,
                total_messages  INTEGER DEFAULT 0,
                kb_hits         INTEGER DEFAULT 0,
                kb_misses       INTEGER DEFAULT 0,
                avg_response_ms REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_used_kb
                ON conversations(used_kb);
        """)
        self.conn.commit()

    def log_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[str],
        top_score: float | None,
        response_ms: int,
        error: str | None = None,
    ) -> None:
        """Registra una interacción en la base de datos.

        No propaga excepciones: el logging nunca debe romper el agente.
        """
        try:
            now = datetime.now(UTC).isoformat()
            used_kb = 1 if sources else 0
            sources_json = json.dumps(sources, ensure_ascii=False)

            self.conn.execute(
                """
                INSERT INTO conversations
                    (session_id, timestamp, question, answer, sources,
                     top_score, used_kb, response_ms, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, now, question, answer, sources_json, top_score, used_kb, response_ms, error),
            )

            # Upsert sessions: INSERT OR IGNORE + UPDATE para compatibilidad SQLite 3.x
            self.conn.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (session_id, started_at, last_active,
                     total_messages, kb_hits, kb_misses, avg_response_ms)
                VALUES (?, ?, ?, 0, 0, 0, 0.0)
                """,
                (session_id, now, now),
            )

            # Recalcular avg_response_ms incrementalmente
            row = self.conn.execute(
                "SELECT total_messages, avg_response_ms FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            prev_count = row["total_messages"]
            prev_avg = row["avg_response_ms"]
            new_count = prev_count + 1
            new_avg = (prev_avg * prev_count + response_ms) / new_count

            self.conn.execute(
                """
                UPDATE sessions SET
                    last_active     = ?,
                    total_messages  = total_messages + 1,
                    kb_hits         = kb_hits   + ?,
                    kb_misses       = kb_misses + ?,
                    avg_response_ms = ?
                WHERE session_id = ?
                """,
                (now, used_kb, 1 - used_kb, new_avg, session_id),
            )

            self.conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al registrar interacción en ConversationLogger: %s", exc)

    def get_stats(self) -> dict:
        """Retorna métricas globales del sistema."""
        try:
            total_conv = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

            total_sessions = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

            kb_hits = self.conn.execute("SELECT COUNT(*) FROM conversations WHERE used_kb = 1").fetchone()[0]

            kb_hit_rate = (kb_hits / total_conv) if total_conv > 0 else 0.0

            avg_ms_row = self.conn.execute("SELECT AVG(response_ms) FROM conversations").fetchone()[0]
            avg_response_ms = avg_ms_row if avg_ms_row is not None else 0.0

            unanswered = self.conn.execute("""
                SELECT question
                FROM conversations
                WHERE used_kb = 0
                ORDER BY timestamp DESC
                LIMIT 10
                """).fetchall()
            top_unanswered = [row["question"] for row in unanswered]

            return {
                "total_conversations": total_conv,
                "total_sessions": total_sessions,
                "kb_hit_rate": kb_hit_rate,
                "avg_response_ms": avg_response_ms,
                "top_unanswered_topics": top_unanswered,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al obtener stats: %s", exc)
            return {
                "total_conversations": 0,
                "total_sessions": 0,
                "kb_hit_rate": 0.0,
                "avg_response_ms": 0.0,
                "top_unanswered_topics": [],
            }

    def get_recent_conversations(self, limit: int = 20) -> list[dict]:
        """Retorna las últimas N conversaciones ordenadas por timestamp desc."""
        try:
            rows = self.conn.execute(
                """
                SELECT id, session_id, timestamp, question, answer,
                       sources, top_score, used_kb, response_ms, error
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            result = []
            for row in rows:
                record = dict(row)
                sources_raw = record.get("sources")
                if isinstance(sources_raw, str):
                    try:
                        record["sources"] = json.loads(sources_raw)
                    except (json.JSONDecodeError, ValueError):
                        record["sources"] = []
                result.append(record)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al obtener conversaciones recientes: %s", exc)
            return []

    def get_coverage_gaps(self, limit: int = 10) -> list[dict]:
        """Retorna preguntas donde used_kb=0 ordenadas por frecuencia.

        Identifica gaps de cobertura: temas que los usuarios preguntan
        pero para los que no hay artículos en la KB.
        """
        try:
            rows = self.conn.execute(
                """
                SELECT
                    question,
                    COUNT(*) AS count,
                    MAX(timestamp) AS last_asked
                FROM conversations
                WHERE used_kb = 0
                GROUP BY question
                ORDER BY count DESC, last_asked DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "question": row["question"],
                    "count": row["count"],
                    "last_asked": row["last_asked"],
                }
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al obtener coverage gaps: %s", exc)
            return []

    def close(self) -> None:
        """Cierra la conexión SQLite."""
        self.conn.close()
