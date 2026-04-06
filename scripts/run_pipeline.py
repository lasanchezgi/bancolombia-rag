"""
Script CLI para ejecutar el pipeline de limpieza, chunking e indexación.

Lee documentos crudos desde data/raw/, aplica Cleaner y Chunker, genera
embeddings con Embedder y los indexa en ChromaDB a través de ChromaRepository.

Uso:
    uv run python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings.embedder import Embedder
from src.pipeline.chunker import Chunker
from src.pipeline.cleaner import Cleaner
from src.scraper.storage import ScraperStorage
from src.vector_store.chroma_repository import ChromaRepository

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Ejecuta el pipeline completo: limpieza → chunking → embedding → indexación."""
    load_dotenv()

    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    chroma_host: str = os.getenv("CHROMA_HOST", "local")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "bancolombia_kb")

    storage = ScraperStorage(Path("data/raw"))
    cleaner = Cleaner()
    chunker = Chunker()
    embedder = Embedder(api_key=openai_api_key)
    repository = ChromaRepository(host=chroma_host, port=chroma_port, collection_name=collection_name)

    raw_pages = storage.load_all()
    logger.info("Cargadas %d páginas desde data/raw/", len(raw_pages))

    all_chunks: list[dict] = []
    empty_pages = 0

    for page in raw_pages:
        cleaned = cleaner.clean(page)
        if cleaned is None:
            logger.info("✗ [vacía] %s", page.get("url", "?"))
            empty_pages += 1
            continue

        chunks = chunker.chunk(cleaned)
        if not chunks:
            logger.info("✗ [sin chunks] %s", page.get("url", "?"))
            empty_pages += 1
            continue

        all_chunks.extend(chunks)

    logger.info(
        "Pipeline: %d páginas → %d chunks totales (%d páginas vacías)",
        len(raw_pages),
        len(all_chunks),
        empty_pages,
    )

    if not all_chunks:
        print("\nNo hay chunks para indexar. Verifica data/raw/.")
        return

    logger.info("Generando embeddings para %d chunks...", len(all_chunks))
    embedded_chunks = embedder.embed_chunks(all_chunks)

    logger.info("Indexando en ChromaDB...")
    repository.add_documents(embedded_chunks)

    total = repository.count()
    logger.info("Documentos en colección tras indexación: %d", total)

    category_counts: Counter[str] = Counter(c["category"] for c in all_chunks)
    categories_str = ", ".join(f"{cat}: {n}" for cat, n in sorted(category_counts.items()))

    print(
        f"\n✅ Pipeline completo: {len(all_chunks)} chunks indexados en ChromaDB\n"
        f"Categorías: {{{categories_str}}}\n"
        f"Colección: {collection_name} | Documentos: {total}"
    )


if __name__ == "__main__":
    main()
