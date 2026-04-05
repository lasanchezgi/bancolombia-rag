"""
Vector store package: puerto abstracto e implementación ChromaDB.

VectorStoreRepository define la interfaz (puerto) que cualquier backend
de vector store debe implementar. ChromaRepository es el adaptador concreto
para ChromaDB, que puede sustituirse sin cambiar la lógica de negocio.
"""

from .chroma_repository import ChromaRepository
from .repository import VectorStoreRepository

__all__ = ["VectorStoreRepository", "ChromaRepository"]
