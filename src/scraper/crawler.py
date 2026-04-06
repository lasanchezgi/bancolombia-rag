"""
Crawler asíncrono para páginas públicas de Bancolombia.

Estrategia sitemap-driven: descarga sitemap-personas.xml, extrae todas
las <loc> URLs, filtra las prohibidas por robots.txt y crawlea las páginas
con httpx.AsyncClient respetando un delay de cortesía y concurrencia máxima.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Patrones de URL explícitamente excluidos por robots.txt de Bancolombia
_BLOCKED_PATTERNS: tuple[str, ...] = (
    "solicitud-de-productos",
    "buscador",
    "formulario-preaprobados",
    "prueba-script",
    "banco-script",
    "preaprobados",
    "!ut/",
    "formulario",
    "ampliar-plazo",
    "temp-test",
    "test-banner",
    "-old",
)

_USER_AGENT = "ClaudeBot/1.0"
_REQUEST_TIMEOUT = 10.0
_DELAY_BETWEEN_REQUESTS = 0.5
_MAX_CONCURRENCY = 5


class Crawler:
    """Crawlea páginas de bancolombia.com a partir del sitemap XML.

    Attributes:
        base_url: URL raíz del sitio (usada solo como referencia; las URLs
                  reales vienen del sitemap).
        max_pages: Número máximo de páginas a procesar.
    """

    def __init__(self, base_url: str, max_pages: int) -> None:
        """Inicializa el Crawler.

        Args:
            base_url: URL raíz del sitio (e.g. ``https://www.bancolombia.com``).
            max_pages: Máximo de páginas a crawlear.
        """
        self.base_url = base_url
        self.max_pages = max_pages

    def fetch_sitemap_urls(self, sitemap_url: str) -> list[str]:
        """Descarga y parsea el sitemap XML, devolviendo URLs permitidas.

        Descarga el sitemap de forma síncrona (se llama una sola vez antes
        del loop async), extrae todas las etiquetas ``<loc>`` y filtra
        las URLs que coincidan con los patrones bloqueados por robots.txt.

        Args:
            sitemap_url: URL absoluta del archivo sitemap XML.

        Returns:
            Lista de URLs absolutas limpias y crawleables.

        Raises:
            httpx.HTTPError: Si no se puede descargar el sitemap.
        """
        response = httpx.get(
            sitemap_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        all_urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

        allowed = [url for url in all_urls if not self._is_blocked(url)]
        logger.info("Sitemap: %d URLs totales → %d permitidas", len(all_urls), len(allowed))
        return allowed

    async def fetch_page(self, url: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Descarga el HTML de una URL con manejo defensivo de errores.

        Args:
            url: URL absoluta de la página a descargar.
            client: Instancia de ``httpx.AsyncClient`` ya configurada.

        Returns:
            Dict ``{"url": str, "html": str, "status_code": int}`` si la
            descarga fue exitosa, o ``None`` si ocurrió cualquier error.
        """
        try:
            response = await client.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            return {
                "url": url,
                "html": response.text,
                "status_code": response.status_code,
            }
        except httpx.TimeoutException:
            logger.warning("Timeout al descargar: %s", url)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %d en: %s", exc.response.status_code, url)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error inesperado en %s: %s", url, exc)
            return None

    async def crawl(self, sitemap_url: str, max_pages: int) -> AsyncGenerator[dict[str, Any], None]:
        """Crawlea páginas del sitemap y genera dicts de páginas descargadas.

        Obtiene URLs del sitemap, limita a ``max_pages``, y descarga cada
        página con concurrencia máxima de 5 y un delay de 0.5 s entre requests.

        Args:
            sitemap_url: URL del sitemap XML desde el que obtener las URLs.
            max_pages: Número máximo de páginas a intentar descargar.

        Yields:
            Dict ``{"url", "html", "status_code"}`` por cada página exitosa.
        """
        urls = self.fetch_sitemap_urls(sitemap_url)[:max_pages]
        total = len(urls)
        logger.info("Iniciando crawl de %d URLs (límite: %d)", total, max_pages)

        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
        headers = {"User-Agent": _USER_AGENT}
        success_count = 0

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            for i, url in enumerate(urls, start=1):
                async with semaphore:
                    page = await self.fetch_page(url, client)
                    if page is not None:
                        success_count += 1
                        yield page
                    if i < total:
                        await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

        logger.info(
            "Crawl finalizado: %d/%d páginas descargadas exitosamente",
            success_count,
            total,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_blocked(url: str) -> bool:
        """Indica si una URL debe excluirse según los patrones de robots.txt.

        Args:
            url: URL absoluta a evaluar.

        Returns:
            ``True`` si la URL contiene algún patrón bloqueado.
        """
        return any(pattern in url for pattern in _BLOCKED_PATTERNS)
