"""
Script CLI para ejecutar el pipeline de limpieza, chunking e indexación.

Lee documentos crudos desde data/raw/, aplica Cleaner y Chunker, genera
embeddings con Embedder y los indexa en ChromaDB a través de ChromaRepository.

Uso:
    uv run python scripts/run_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings.embedder import Embedder
from src.pipeline.chunker import Chunker
from src.pipeline.cleaner import Cleaner
from src.scraper.storage import ScraperStorage
from src.vector_store.chroma_repository import ChromaRepository


def main() -> None:
    """Ejecuta el pipeline completo: limpieza → chunking → embedding → indexación.

    Lee CHUNK_SIZE y CHUNK_OVERLAP del entorno, carga todos los documentos
    crudos, los limpia, los divide en chunks, genera sus embeddings y los
    upserta en ChromaDB.
    """
    load_dotenv()

    chunk_size: int = int(os.environ["CHUNK_SIZE"])
    chunk_overlap: int = int(os.environ["CHUNK_OVERLAP"])
    raw_dir = Path("data/raw")

    raise NotImplementedError


if __name__ == "__main__":
    main()
