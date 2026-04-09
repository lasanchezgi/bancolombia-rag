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
from typing import Any

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

            CREATE TABLE IF NOT EXISTS mcp_calls (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id      INTEGER,
                session_id           TEXT NOT NULL,
                timestamp            TEXT NOT NULL,
                tool_name            TEXT NOT NULL,
                query_sent           TEXT,
                category_filter      TEXT,
                use_reranking        INTEGER,
                chunks_retrieved     INTEGER,
                chunks_after_rerank  INTEGER,
                top_chromadb_score   REAL,
                top_rerank_score     REAL,
                urls_returned        TEXT,
                titles_returned      TEXT,
                retrieval_ms         INTEGER,
                reranking_ms         INTEGER,
                total_ms             INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_used_kb
                ON conversations(used_kb);
            CREATE INDEX IF NOT EXISTS idx_mcp_calls_session
                ON mcp_calls(session_id);
            CREATE INDEX IF NOT EXISTS idx_mcp_calls_tool
                ON mcp_calls(tool_name);
            CREATE INDEX IF NOT EXISTS idx_mcp_calls_timestamp
                ON mcp_calls(timestamp);
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
    ) -> int:
        """Registra una interacción en la base de datos.

        No propaga excepciones: el logging nunca debe romper el agente.
        Retorna el id del registro creado, o -1 si falla.
        """
        try:
            now = datetime.now(UTC).isoformat()
            used_kb = 1 if sources else 0
            sources_json = json.dumps(sources, ensure_ascii=False)

            cursor = self.conn.execute(
                """
                INSERT INTO conversations
                    (session_id, timestamp, question, answer, sources,
                     top_score, used_kb, response_ms, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, now, question, answer, sources_json, top_score, used_kb, response_ms, error),
            )
            conversation_id: int = cursor.lastrowid  # type: ignore[assignment]

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
            return conversation_id
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al registrar interacción en ConversationLogger: %s", exc)
            return -1

    def log_mcp_call(
        self,
        session_id: str,
        tool_name: str,
        query_sent: str | None = None,
        category_filter: str | None = None,
        use_reranking: bool = True,
        chunks_retrieved: int = 0,
        chunks_after_rerank: int = 0,
        top_chromadb_score: float | None = None,
        top_rerank_score: float | None = None,
        urls_returned: list[str] | None = None,
        titles_returned: list[str] | None = None,
        retrieval_ms: int = 0,
        reranking_ms: int = 0,
        total_ms: int = 0,
        conversation_id: int | None = None,
    ) -> int:
        """Registra una llamada al servidor MCP con trazabilidad completa.

        Retorna el id del registro creado, o -1 si falla.
        urls_returned y titles_returned se serializan como JSON.
        Si falla: loguear error pero NO propagar excepción.
        """
        try:
            now = datetime.now(UTC).isoformat()
            urls_json = json.dumps(urls_returned or [], ensure_ascii=False)
            titles_json = json.dumps(titles_returned or [], ensure_ascii=False)
            use_reranking_int = 1 if use_reranking else 0

            cursor = self.conn.execute(
                """
                INSERT INTO mcp_calls (
                    conversation_id, session_id, timestamp, tool_name,
                    query_sent, category_filter, use_reranking,
                    chunks_retrieved, chunks_after_rerank,
                    top_chromadb_score, top_rerank_score,
                    urls_returned, titles_returned,
                    retrieval_ms, reranking_ms, total_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    session_id,
                    now,
                    tool_name,
                    query_sent,
                    category_filter,
                    use_reranking_int,
                    chunks_retrieved,
                    chunks_after_rerank,
                    top_chromadb_score,
                    top_rerank_score,
                    urls_json,
                    titles_json,
                    retrieval_ms,
                    reranking_ms,
                    total_ms,
                ),
            )
            mcp_call_id: int = cursor.lastrowid  # type: ignore[assignment]
            self.conn.commit()
            return mcp_call_id
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al registrar llamada MCP en ConversationLogger: %s", exc)
            return -1

    def _link_mcp_calls_to_conversation(self, mcp_call_ids: list[int], conversation_id: int) -> None:
        """Actualiza los mcp_calls con el conversation_id tras conocerlo."""
        if not mcp_call_ids:
            return
        try:
            placeholders = ",".join("?" * len(mcp_call_ids))
            self.conn.execute(
                f"UPDATE mcp_calls SET conversation_id = ? WHERE id IN ({placeholders})",  # noqa: S608
                [conversation_id, *mcp_call_ids],
            )
            self.conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al vincular mcp_calls con conversation_id: %s", exc)

    def get_mcp_traces(
        self,
        session_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retorna llamadas MCP con filtros opcionales.

        Parsea urls_returned y titles_returned de JSON a list.
        Ordenado por timestamp desc.
        """
        try:
            conditions: list[str] = []
            params: list[object] = []
            if session_id is not None:
                conditions.append("session_id = ?")
                params.append(session_id)
            if tool_name is not None:
                conditions.append("tool_name = ?")
                params.append(tool_name)

            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)

            rows = self.conn.execute(
                f"""
                SELECT * FROM mcp_calls
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
                """,  # noqa: S608
                params,
            ).fetchall()

            result = []
            for row in rows:
                record = dict(row)
                for field in ("urls_returned", "titles_returned"):
                    raw = record.get(field)
                    if isinstance(raw, str):
                        try:
                            record[field] = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            record[field] = []
                result.append(record)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al obtener trazas MCP: %s", exc)
            return []

    def get_rag_performance_metrics(self) -> dict[str, Any]:
        """Retorna métricas agregadas del pipeline RAG."""
        try:
            row = self.conn.execute("""
                SELECT
                    AVG(top_chromadb_score)  AS avg_chromadb_score,
                    AVG(top_rerank_score)    AS avg_rerank_score,
                    AVG(retrieval_ms)        AS avg_retrieval_ms,
                    AVG(reranking_ms)        AS avg_reranking_ms,
                    AVG(total_ms)            AS avg_total_ms,
                    COUNT(*)                 AS total_mcp_calls,
                    SUM(use_reranking)       AS calls_with_reranking
                FROM mcp_calls
                """).fetchone()

            avg_chromadb = row["avg_chromadb_score"] or 0.0
            avg_rerank = row["avg_rerank_score"] or 0.0
            total = row["total_mcp_calls"] or 0
            with_reranking = row["calls_with_reranking"] or 0

            # Reranking improvement: avg(rerank_score - chromadb_score) where both exist
            improvement_row = self.conn.execute("""
                SELECT AVG(top_rerank_score - top_chromadb_score) AS improvement
                FROM mcp_calls
                WHERE top_rerank_score IS NOT NULL
                  AND top_chromadb_score IS NOT NULL
                """).fetchone()
            reranking_improvement = improvement_row["improvement"] or 0.0

            # Most frequent queries (top 10)
            freq_rows = self.conn.execute("""
                SELECT query_sent, COUNT(*) AS count
                FROM mcp_calls
                WHERE query_sent IS NOT NULL
                GROUP BY query_sent
                ORDER BY count DESC
                LIMIT 10
                """).fetchall()
            most_frequent_queries = [{"query": r["query_sent"], "count": r["count"]} for r in freq_rows]

            # Low score queries: top_chromadb_score < 0.5 (top 10)
            low_rows = self.conn.execute("""
                SELECT query_sent, top_chromadb_score, timestamp
                FROM mcp_calls
                WHERE top_chromadb_score IS NOT NULL
                  AND top_chromadb_score < 0.5
                ORDER BY top_chromadb_score ASC
                LIMIT 10
                """).fetchall()
            low_score_queries = [
                {
                    "query": r["query_sent"],
                    "top_chromadb_score": r["top_chromadb_score"],
                    "timestamp": r["timestamp"],
                }
                for r in low_rows
            ]

            return {
                "avg_chromadb_score": avg_chromadb,
                "avg_rerank_score": avg_rerank,
                "reranking_improvement": reranking_improvement,
                "avg_retrieval_ms": row["avg_retrieval_ms"] or 0.0,
                "avg_reranking_ms": row["avg_reranking_ms"] or 0.0,
                "avg_total_ms": row["avg_total_ms"] or 0.0,
                "total_mcp_calls": total,
                "calls_with_reranking": with_reranking,
                "calls_without_reranking": total - with_reranking,
                "most_frequent_queries": most_frequent_queries,
                "low_score_queries": low_score_queries,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error al obtener métricas RAG: %s", exc)
            return {
                "avg_chromadb_score": 0.0,
                "avg_rerank_score": 0.0,
                "reranking_improvement": 0.0,
                "avg_retrieval_ms": 0.0,
                "avg_reranking_ms": 0.0,
                "avg_total_ms": 0.0,
                "total_mcp_calls": 0,
                "calls_with_reranking": 0,
                "calls_without_reranking": 0,
                "most_frequent_queries": [],
                "low_score_queries": [],
            }

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
