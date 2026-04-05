"""
Script CLI para ejecutar el pipeline completo de scraping.

Lee la configuración desde variables de entorno (o .env), instancia
Crawler, Parser y ScraperStorage, y ejecuta el ciclo de crawl-parse-save
guardando cada página en data/raw/ como archivo JSON.

Uso:
    uv run python scripts/run_scraper.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from src.scraper.crawler import Crawler
from src.scraper.parser import Parser
from src.scraper.storage import ScraperStorage


async def main() -> None:
    """Ejecuta el pipeline de scraping de extremo a extremo.

    Lee BANCOLOMBIA_BASE_URL y MAX_PAGES del entorno, crawlea las páginas,
    las parsea y guarda los documentos resultantes en data/raw/.
    """
    load_dotenv()

    base_url: str = os.environ["BANCOLOMBIA_BASE_URL"]
    max_pages: int = int(os.environ["MAX_PAGES"])
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError


if __name__ == "__main__":
    asyncio.run(main())
