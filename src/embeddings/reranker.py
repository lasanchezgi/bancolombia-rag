import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None  # type: ignore[assignment,misc]

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Reranker de dos etapas usando Cross-Encoder.

    Pipeline:
    1. ChromaDB recupera top-N candidatos por similitud coseno
       (rápido pero aproximado)
    2. Cross-Encoder puntúa cada par (query, documento)
       conjuntamente (más lento pero preciso)
    3. Se reordenan y retornan los top-K mejores

    Esto mejora la precisión del retrieval sin reemplazar
    ChromaDB — lo complementa.

    Si sentence-transformers no está instalado, retorna los
    documentos sin reordenar (fallback graceful).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        if not RERANKER_AVAILABLE:
            logger.warning(
                "sentence-transformers no instalado — reranker desactivado. " "Instalar con: uv sync --extra reranking"
            )
            self.model = None
            return
        self.model = CrossEncoder(model_name)
        logger.info("Reranker inicializado: %s", model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reordena documentos por relevancia real con el query.

        Args:
            query: consulta original del usuario
            documents: lista de dicts con al menos "text" y
                       los metadatos retornados por ChromaDB
            top_k: cuántos documentos retornar tras el reranking

        Returns:
            Lista de top_k documentos reordenados, cada uno con
            campo adicional "rerank_score": float
        """
        if not documents:
            return []
        if self.model is None:
            logger.warning("Reranker no disponible — retornando documentos sin reordenar")
            return documents[:top_k]
        try:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = self.model.predict(pairs)
            logger.info("Reranking: %d → %d docs", len(documents), top_k)
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            sorted_docs = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
            return sorted_docs[:top_k]
        except Exception as exc:
            logger.error("Error en reranking: %s — usando fallback", exc)
            for doc in documents:
                doc["rerank_score"] = None
            return documents[:top_k]
