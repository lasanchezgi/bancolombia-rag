"""
Parser HTML que extrae texto limpio y metadatos de páginas crudas.

Usa BeautifulSoup4 para eliminar scripts, estilos y boilerplate, devolviendo
un dict estructurado con url, título, texto plano y metadatos básicos.
El dict resultante es el formato canónico que consume ScraperStorage.
"""

from __future__ import annotations

from bs4 import BeautifulSoup


class Parser:
    """Extrae contenido estructurado a partir de HTML crudo.

    Elimina elementos no informativos (scripts, estilos, nav, footer)
    antes de devolver el texto principal de la página.
    """

    # Tags HTML que se eliminan antes de extraer el texto
    NOISE_TAGS: tuple[str, ...] = (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
    )

    def parse(self, url: str, html: str) -> dict[str, str]:
        """Parsea HTML crudo y devuelve un documento estructurado.

        Args:
            url: URL de origen de la página.
            html: Contenido HTML crudo descargado por el Crawler.

        Returns:
            Dict con las siguientes claves:
            - ``url``: URL de origen.
            - ``title``: Título de la página (etiqueta ``<title>``).
            - ``text``: Texto plano limpio, sin HTML ni boilerplate.
            - ``metadata``: Dict con información adicional (e.g. description).
        """
        raise NotImplementedError

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrae el título de la página desde la etiqueta <title>.

        Args:
            soup: Árbol BeautifulSoup de la página.

        Returns:
            Título como string, o cadena vacía si no existe.
        """
        raise NotImplementedError

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extrae metadatos de las etiquetas <meta> de la página.

        Args:
            soup: Árbol BeautifulSoup de la página.

        Returns:
            Dict con pares clave-valor de metadatos (e.g. ``description``).
        """
        raise NotImplementedError
