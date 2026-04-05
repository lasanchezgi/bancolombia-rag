"""
Embeddings package: convierte chunks de texto en vectores numéricos.

Usa la API de OpenAI (modelo text-embedding-3-small) para generar
representaciones densas de los fragmentos de texto.
"""

from .embedder import Embedder

__all__ = ["Embedder"]
