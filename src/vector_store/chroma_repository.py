"""
Adaptador ChromaDB para VectorStoreRepository.

Implementa el puerto VectorStoreRepository usando ChromaDB en modo dual:
- ``host == "local"``: PersistentClient en ``.chroma/`` (desarrollo local).
- Cualquier otro host: HttpClient para ChromaDB en Docker.

La colección se crea con métrica coseno (``hnsw:space: cosine``) para que
las distancias devueltas sean directamente 1 - similaridad.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from .repository import VectorStoreRepository

logger = logging.getLogger(__name__)

_METADATA_KEYS = (
    "url",
    "title",
    "category",
    "subcategory",
    "extraction_date",
    "chunk_index",
    "total_chunks",
    "word_count",
)


class ChromaRepository(VectorStoreRepository):
    """Implementación de VectorStoreRepository respaldada por ChromaDB.

    Soporta modo local (PersistentClient) y modo remoto (HttpClient).

    Attributes:
        collection_name: Nombre de la colección activa en ChromaDB.
        client: Cliente ChromaDB (Persistent o HTTP).
        collection: Colección ChromaDB activa.
    """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        """Inicializa la conexión a ChromaDB y obtiene/crea la colección.

        Args:
            host: Host del servidor ChromaDB, o ``"local"`` para modo persistente.
            port: Puerto del servidor ChromaDB (ignorado en modo local).
            collection_name: Nombre de la colección a usar o crear.
        """
        self.collection_name = collection_name

        if host == "local":
            self.client = chromadb.PersistentClient(path=".chroma")
            logger.info("ChromaDB: modo PersistentClient (path=.chroma)")
        else:
            self.client = chromadb.HttpClient(host=host, port=port)
            logger.info("ChromaDB: modo HttpClient (%s:%d)", host, port)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        """Upserta chunks enriquecidos con embeddings en la colección ChromaDB.

        Args:
            chunks: Lista de dicts con al menos ``chunk_id``, ``text`` y
                    ``embedding`` (salida de ``Embedder.embed_chunks()``).
        """
        ids = [c["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{k: c.get(k, "") for k in _METADATA_KEYS} for c in chunks]

        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info("Upserted %d documents to ChromaDB collection '%s'", len(chunks), self.collection_name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Consulta ChromaDB por vecinos más cercanos.

        Args:
            query_embedding: Vector de la consulta.
            top_k: Número de resultados a retornar.
            filters: Filtros de metadatos para ChromaDB ``where`` clause.

        Returns:
            Lista de dicts con ``chunk_id``, ``text``, ``url``, ``title``,
            ``category``, ``score`` (1 - distancia coseno) y ``chunk_index``.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "score": 1.0 - distance,
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )

        return output

    def delete_collection(self) -> None:
        """Elimina la colección completa de ChromaDB."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted ChromaDB collection '%s'", self.collection_name)

    def count(self) -> int:
        """Retorna el número de documentos en la colección.

        Returns:
            Conteo de documentos almacenados.
        """
        return self.collection.count()
