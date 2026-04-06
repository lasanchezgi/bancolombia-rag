"""
Interfaz abstracta (puerto) para backends de vector store.

Define el contrato que toda implementación concreta de vector store debe
satisfacer. Siguiendo los principios de arquitectura hexagonal, el resto
del sistema depende únicamente de esta interfaz, no de ChromaDB ni de
ningún otro proveedor específico.

Para agregar un nuevo backend (Pinecone, Qdrant, Weaviate...), basta con
crear una subclase que implemente los cuatro métodos abstractos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreRepository(ABC):
    """Puerto abstracto para operaciones sobre un vector store.

    Todas las implementaciones concretas deben heredar de esta clase
    e implementar todos los métodos marcados como ``@abstractmethod``.
    """

    @abstractmethod
    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        """Inserta o actualiza documentos con sus embeddings en el store.

        Args:
            chunks: Lista de dicts de chunks enriquecidos con el campo
                    ``"embedding"`` (salida de ``Embedder.embed_chunks()``).
                    Cada dict debe tener al menos: ``chunk_id``, ``text``,
                    ``embedding``, y los metadatos relevantes.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recupera los documentos más similares a un vector de consulta.

        Args:
            query_embedding: Vector de embedding de la consulta del usuario.
            top_k: Número de resultados a devolver.
            filters: Filtros de metadatos opcionales (e.g. ``{"category": "cuentas"}``).

        Returns:
            Lista de dicts ordenados por similitud descendente, cada uno con:
            ``chunk_id``, ``text``, ``url``, ``title``, ``category``,
            ``score`` (1 - distancia coseno) y ``chunk_index``.
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Elimina toda la colección del vector store.

        Útil para re-indexar desde cero sin datos residuales.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Devuelve el número de documentos almacenados actualmente.

        Returns:
            Entero con el total de documentos en la colección.
        """
        ...
