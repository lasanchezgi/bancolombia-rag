"""
Crawler asíncrono para páginas públicas de Bancolombia.

Utiliza httpx.AsyncClient para descargar páginas respetando el límite MAX_PAGES.
Descubre enlaces internos a partir de la URL base y los encola para visitar.
Genera tuplas (url, html) que se pasan al Parser para extracción de contenido.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx


class Crawler:
    """Crawlea páginas de forma asíncrona a partir de una URL base.

    Attributes:
        base_url: URL raíz desde la que comienza el crawling.
        max_pages: Número máximo de páginas a visitar antes de detenerse.
    """

    def __init__(self, base_url: str, max_pages: int) -> None:
        """Inicializa el Crawler con la URL base y el límite de páginas.

        Args:
            base_url: URL raíz del sitio a crawlear.
            max_pages: Máximo número de páginas únicas a visitar.
        """
        self.base_url = base_url
        self.max_pages = max_pages

    async def crawl(self) -> AsyncGenerator[tuple[str, str], None]:
        """Crawlea páginas y genera tuplas (url, html).

        Visita la URL base, descubre enlaces internos y los sigue hasta
        alcanzar max_pages. Solo visita URLs del mismo dominio que base_url.

        Yields:
            Tupla ``(page_url, raw_html)`` por cada página descargada con éxito.

        Raises:
            httpx.HTTPError: Si una petición falla de forma no recuperable.
        """
        raise NotImplementedError

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> str:
        """Descarga el HTML de una URL usando el cliente httpx proporcionado.

        Args:
            client: Instancia de httpx.AsyncClient ya configurada.
            url: URL absoluta de la página a descargar.

        Returns:
            Contenido HTML crudo como string.

        Raises:
            httpx.HTTPStatusError: Si el servidor devuelve un código 4xx/5xx.
        """
        raise NotImplementedError

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extrae enlaces internos desde el HTML de una página.

        Args:
            html: Contenido HTML de la página actual.
            base_url: URL base para resolver rutas relativas.

        Returns:
            Lista de URLs absolutas internas encontradas en la página.
        """
        raise NotImplementedError
