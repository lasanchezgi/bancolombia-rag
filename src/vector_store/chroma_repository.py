"""
Adaptador ChromaDB para VectorStoreRepository.

Implementa el puerto VectorStoreRepository usando el cliente HTTP de ChromaDB,
lo que permite conectar tanto a un servidor local como a uno dentro de Docker
sin modificar el código de negocio. La conexión se configura a través de las
variables de entorno CHROMA_HOST y CHROMA_PORT.
"""

from __future__ import annotations

import os

import chromadb

from .repository import VectorStoreRepository


class ChromaRepository(VectorStoreRepository):
    """Implementación de VectorStoreRepository respaldada por ChromaDB.

    Usa el HttpClient de ChromaDB para conectar a un servidor externo
    (self-hosted), lo que garantiza compatibilidad con Docker Compose.

    Attributes:
        client: Cliente HTTP de ChromaDB.
        collection: Colección activa dentro de ChromaDB.
    """

    DEFAULT_COLLECTION = "bancolombia"

    def __init__(self, collection_name: str = DEFAULT_COLLECTION) -> None:
        """Inicializa la conexión a ChromaDB y obtiene/crea la colección.

        La configuración de host y puerto se lee de las variables de entorno
        ``CHROMA_HOST`` (default: ``localhost``) y ``CHROMA_PORT`` (default: ``8000``).

        Args:
            collection_name: Nombre de la colección a usar o crear en ChromaDB.
        """
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upserta documentos en la colección ChromaDB.

        Args:
            ids: Identificadores únicos de los documentos.
            embeddings: Vectores pre-computados.
            documents: Texto plano de cada chunk.
            metadatas: Metadatos por documento.
        """
        raise NotImplementedError

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Consulta ChromaDB por vecinos más cercanos.

        Args:
            query_embedding: Vector de la consulta del usuario.
            top_k: Número de resultados a retornar.

        Returns:
            Lista de dicts con ``id``, ``document``, ``metadata``, ``distance``.
        """
        raise NotImplementedError

    def delete_collection(self) -> None:
        """Elimina la colección completa de ChromaDB."""
        raise NotImplementedError

    def count(self) -> int:
        """Retorna el número de documentos en la colección ChromaDB.

        Returns:
            Conteo de documentos almacenados.
        """
        raise NotImplementedError
