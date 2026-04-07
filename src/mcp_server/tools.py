"""
Definición de herramientas MCP para el pipeline de retrieval RAG.

Las tres tools (search_knowledge_base, get_article_by_url, list_categories)
y el resource (knowledgebase://stats) encapsulan el acceso al vector store
de ChromaDB. Cada tool maneja sus propios errores y nunca lanza excepciones
al agente, devolviendo en su lugar un dict con la clave "error".
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any

from fastmcp import FastMCP

from src.embeddings.embedder import Embedder
from src.embeddings.reranker import Reranker
from src.vector_store.chroma_repository import ChromaRepository

logger = logging.getLogger(__name__)

_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """Retorna el singleton del Reranker, instanciándolo la primera vez."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def _get_repository() -> ChromaRepository:
    """Instancia ChromaRepository con config del entorno."""
    host = os.getenv("CHROMA_HOST", "local")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    collection = os.getenv("CHROMA_COLLECTION", "bancolombia_kb")
    return ChromaRepository(host=host, port=port, collection_name=collection)


def _get_embedder() -> Embedder:
    """Instancia Embedder con config del entorno."""
    api_key = os.environ["OPENAI_API_KEY"]
    return Embedder(api_key=api_key)


def register_tools(mcp: FastMCP) -> None:
    """Registra las 3 tools y el resource en la instancia FastMCP.

    Args:
        mcp: Instancia de FastMCP sobre la que se registran las herramientas.
    """

    @mcp.tool  # type: ignore[misc]
    def search_knowledge_base(
        query: str,
        top_k: int = 5,
        category: str | None = None,
        use_reranking: bool = True,
    ) -> dict[str, Any]:
        """Busca información sobre productos, servicios y contenido de Bancolombia en la base de conocimiento.

        Usa esta herramienta cuando el usuario pregunte sobre cuentas, tarjetas, créditos,
        beneficios, giros, o cualquier producto o servicio bancario.
        Retorna los fragmentos más relevantes con sus fuentes (URLs).

        Args:
            query: Pregunta o consulta en lenguaje natural del usuario
            top_k: Número de resultados a retornar (default: 5, max: 10)
            category: Filtrar por categoría específica. Categorías disponibles: cuentas,
                      tarjetas-de-credito, tarjetas-debito, creditos, beneficios, giros, general. Opcional.
            use_reranking: Si True (default), aplica reranking con Cross-Encoder tras recuperar candidatos de
                           ChromaDB, mejorando la precisión a costa de ~200ms adicionales. Usar False para
                           respuestas rápidas cuando la velocidad es prioritaria.
        """
        try:
            embedder = _get_embedder()
            repository = _get_repository()

            embeddings = embedder.embed_texts([query])
            query_embedding = embeddings[0]

            n_candidates = top_k * 3 if use_reranking else top_k
            filters = {"category": {"$eq": category}} if category else None
            raw_results = repository.query(
                query_embedding=query_embedding,
                top_k=min(n_candidates, 30),
                filters=filters,
            )

            if not raw_results:
                return {"results": [], "total": 0, "message": "No se encontró información relevante", "query": query}

            if use_reranking:
                results = get_reranker().rerank(query, raw_results, top_k)
                retrieval_method = "chromadb+reranking"
            else:
                results = raw_results[:top_k]
                for r in results:
                    r["rerank_score"] = None
                retrieval_method = "chromadb_only"

            return {
                "results": [
                    {
                        "text": r["text"],
                        "url": r["url"],
                        "title": r["title"],
                        "category": r["category"],
                        "relevance_score": r["score"],
                        "chunk_index": r["chunk_index"],
                        "rerank_score": r.get("rerank_score"),
                        "retrieval_method": retrieval_method,
                    }
                    for r in results
                ],
                "total": len(results),
                "query": query,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en search_knowledge_base: %s", exc)
            return {"error": str(exc), "results": [], "total": 0, "query": query}

    @mcp.tool  # type: ignore[misc]
    def get_article_by_url(url: str) -> dict[str, Any]:
        """Recupera el contenido completo de un artículo o página de Bancolombia dado su URL.

        Usa esta herramienta cuando el usuario pida más detalles sobre una página específica,
        o cuando search_knowledge_base retorne una URL relevante y necesites el contenido completo.

        Args:
            url: URL completa de la página de Bancolombia (ej: https://www.bancolombia.com/personas/cuentas/...)
        """
        try:
            repository = _get_repository()

            raw = repository.collection.get(where={"url": {"$eq": url}})
            ids = raw.get("ids", [])
            if not ids:
                return {"found": False, "url": url, "message": "Artículo no encontrado en la base de conocimiento"}

            documents = raw.get("documents", [])
            metadatas = raw.get("metadatas", [])

            # Ordenar chunks por chunk_index
            combined = sorted(zip(metadatas, documents), key=lambda x: x[0].get("chunk_index", 0))

            meta0 = combined[0][0]
            full_text = "\n\n".join(doc for _, doc in combined)

            return {
                "found": True,
                "url": url,
                "title": meta0.get("title", ""),
                "category": meta0.get("category", ""),
                "full_text": full_text,
                "total_chunks": len(combined),
                "extraction_date": meta0.get("extraction_date", ""),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en get_article_by_url: %s", exc)
            return {"error": str(exc), "found": False, "url": url}

    @mcp.tool  # type: ignore[misc]
    def list_categories() -> dict[str, Any]:
        """Lista todas las categorías disponibles en la base de conocimiento de Bancolombia con el número de documentos.

        Usa esta herramienta cuando el usuario pregunte qué temas o productos cubre el asistente,
        o antes de usar search_knowledge_base con filtro de categoría.
        """
        try:
            repository = _get_repository()
            collection_name = os.getenv("CHROMA_COLLECTION", "bancolombia_kb")

            raw = repository.collection.get()
            metadatas = raw.get("metadatas", []) or []

            counts: Counter[str] = Counter(m.get("category", "general") for m in metadatas)

            return {
                "categories": [{"name": cat, "document_count": n} for cat, n in counts.most_common()],
                "total_documents": len(metadatas),
                "collection_name": collection_name,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en list_categories: %s", exc)
            return {"error": str(exc), "categories": [], "total_documents": 0}

    @mcp.resource("knowledgebase://stats")  # type: ignore[misc]
    def get_kb_stats() -> str:
        """Estadísticas de la base de conocimiento de Bancolombia."""
        try:
            repository = _get_repository()
            collection_name = os.getenv("CHROMA_COLLECTION", "bancolombia_kb")
            chroma_host = os.getenv("CHROMA_HOST", "local")

            raw = repository.collection.get()
            metadatas = raw.get("metadatas", []) or []

            counts: Counter[str] = Counter(m.get("category", "general") for m in metadatas)

            extraction_dates = [m.get("extraction_date", "") for m in metadatas if m.get("extraction_date")]
            last_updated = max(extraction_dates) if extraction_dates else ""

            return json.dumps(
                {
                    "total_documents": len(metadatas),
                    "collection_name": collection_name,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "categories": dict(counts),
                    "last_updated": last_updated,
                    "chroma_mode": "local" if chroma_host == "local" else "http",
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en get_kb_stats: %s", exc)
            return json.dumps({"error": str(exc)})
