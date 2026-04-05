"""
Script CLI para ejecutar el pipeline completo de scraping.

Lee la configuración desde variables de entorno (o .env), crawlea el sitemap
de bancolombia.com/personas, parsea cada página y guarda los resultados
en data/raw/ como archivos JSON.

Uso:
    uv run python scripts/run_scraper.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.scraper.crawler import Crawler
from src.scraper.parser import Parser
from src.scraper.storage import ScraperStorage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Ejecuta el pipeline de scraping de extremo a extremo."""
    load_dotenv()

    max_pages: int = int(os.getenv("MAX_PAGES", "80"))
    base_url: str = os.getenv("BANCOLOMBIA_BASE_URL", "https://www.bancolombia.com")
    sitemap_url = f"{base_url.rstrip('/')}/sitemap-personas.xml"

    crawler = Crawler(base_url=base_url, max_pages=max_pages)
    parser = Parser()
    storage = ScraperStorage(Path("data/raw"))

    saved = 0
    empty = 0
    errors = 0

    async for raw_page in crawler.crawl(sitemap_url, max_pages):
        try:
            result = parser.parse(raw_page)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parseando %s: %s", raw_page.get("url"), exc)
            errors += 1
            continue

        if result is None:
            logger.info("✗ [%s] - empty content", raw_page["url"])
            empty += 1
            continue

        storage.save(result)
        saved += 1
        logger.info(
            "✓ [%d/%d] %s/%s (%d chars)",
            saved,
            max_pages,
            result["category"],
            result["title"] or result["url"],
            result["char_count"],
        )

    print(f"\nScraping completo: {saved} páginas guardadas, {empty} vacías, {errors} errores")


if __name__ == "__main__":
    asyncio.run(main())
