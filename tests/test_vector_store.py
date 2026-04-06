"""
Tests unitarios para el paquete vector_store.

Verifica el contrato de la ABC VectorStoreRepository usando un stub
en memoria, sin requerir una instancia real de ChromaDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.vector_store.repository import VectorStoreRepository


class _StubRepository(VectorStoreRepository):
    """Implementación mínima en memoria para verificar el contrato de la ABC."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            self._docs[chunk["chunk_id"]] = chunk

    def query(self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self._docs.values())[:top_k]

    def delete_collection(self) -> None:
        self._docs.clear()

    def count(self) -> int:
        return len(self._docs)


class TestVectorStoreRepository:
    """Tests del contrato de la ABC VectorStoreRepository."""

    def test_abc_cannot_be_instantiated_directly(self) -> None:
        """VectorStoreRepository no puede instanciarse directamente (es abstracta)."""
        with pytest.raises(TypeError):
            VectorStoreRepository()  # type: ignore[abstract]

    def test_stub_satisfies_interface(self) -> None:
        """Un stub que implementa todos los métodos abstractos debe instanciarse."""
        repo = _StubRepository()
        assert repo is not None

    def test_count_starts_at_zero(self) -> None:
        """count() debe devolver 0 en un repositorio recién creado."""
        repo = _StubRepository()
        assert repo.count() == 0

    def test_add_documents_increments_count(self) -> None:
        """add_documents() debe incrementar el conteo de documentos."""
        repo = _StubRepository()
        repo.add_documents([{"chunk_id": "doc-1", "text": "Texto de prueba", "url": "https://example.com"}])
        assert repo.count() == 1

    def test_query_returns_list(self) -> None:
        """query() debe devolver una lista (posiblemente vacía)."""
        repo = _StubRepository()
        result = repo.query(query_embedding=[0.1] * 10, top_k=3)
        assert isinstance(result, list)

    def test_delete_collection_empties_store(self) -> None:
        """delete_collection() debe dejar el repositorio vacío."""
        repo = _StubRepository()
        repo.add_documents([{"chunk_id": "id-1", "text": "doc"}])
        repo.delete_collection()
        assert repo.count() == 0
