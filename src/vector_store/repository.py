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


class VectorStoreRepository(ABC):
    """Puerto abstracto para operaciones sobre un vector store.

    Todas las implementaciones concretas deben heredar de esta clase
    e implementar todos los métodos marcados como ``@abstractmethod``.
    """

    @abstractmethod
    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Inserta o actualiza documentos con sus embeddings en el store.

        Args:
            ids: Identificadores únicos para cada documento.
            embeddings: Vectores de embedding pre-computados.
            documents: Texto plano de cada chunk.
            metadatas: Lista de dicts con metadatos por documento
                       (e.g. ``{"url": "...", "title": "..."}``).
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """Recupera los documentos más similares a un vector de consulta.

        Args:
            query_embedding: Vector de embedding de la consulta del usuario.
            top_k: Número de resultados a devolver.

        Returns:
            Lista de dicts ordenados por similitud descendente, cada uno con:
            - ``id``: Identificador del documento.
            - ``document``: Texto del chunk.
            - ``metadata``: Dict con metadatos del documento.
            - ``distance``: Distancia (menor = más similar).
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
