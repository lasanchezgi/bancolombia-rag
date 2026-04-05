"""
Parser HTML que extrae texto limpio y metadatos de páginas de Bancolombia.

Usa BeautifulSoup4 para eliminar scripts, estilos, navegación y boilerplate
propio de IBM WebSphere Portal, devolviendo un dict canónico con título, texto,
categoría y subcategoría extraídos del path de la URL.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Tags HTML que se eliminan antes de extraer el texto
_NOISE_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "iframe",
    "noscript",
    "aside",
    "form",
)

# Clases CSS de elementos de layout/navegación propios de WebSphere Portal
_NOISE_CLASSES: tuple[str, ...] = (
    "wpthemeFrame",
    "wpthemeComplementaryContent",
    "header-main",
    "footer",
    "menu",
    "breadcrumb",
    "banner",
)

_WHITESPACE_RE = re.compile(r"\s+")


class Parser:
    """Extrae contenido estructurado de páginas HTML de Bancolombia.

    Limpia el árbol DOM eliminando elementos de navegación y boilerplate
    antes de extraer texto, título y metadatos de categorización.
    """

    def parse(self, raw_page: dict[str, Any]) -> dict[str, Any] | None:
        """Parsea una página cruda del crawler y devuelve un documento limpio.

        Args:
            raw_page: Dict con las claves ``"url"`` y ``"html"`` producido
                      por ``Crawler.fetch_page()``.

        Returns:
            Dict con las claves ``url``, ``title``, ``text``, ``category``,
            ``subcategory``, ``extraction_date`` y ``char_count``, o ``None``
            si el texto resultante queda vacío tras la limpieza.
        """
        url: str = raw_page["url"]
        html: str = raw_page["html"]

        soup = BeautifulSoup(html, "html.parser")
        self._remove_noise(soup)

        title = self._extract_title(soup)
        text = self._extract_text(soup)

        if not text:
            return None

        category, subcategory = self._extract_category(url)

        return {
            "url": url,
            "title": title,
            "text": text,
            "category": category,
            "subcategory": subcategory,
            "extraction_date": datetime.now().isoformat(),
            "char_count": len(text),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Elimina tags de ruido y elementos con clases CSS de boilerplate.

        Modifica el árbol ``soup`` in-place.

        Args:
            soup: Árbol BeautifulSoup a limpiar.
        """
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        for css_class in _NOISE_CLASSES:
            for element in soup.find_all(class_=css_class):
                element.decompose()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrae el título: primer <h1>, luego <title>, o cadena vacía.

        Args:
            soup: Árbol BeautifulSoup ya limpio.

        Returns:
            Título de la página como string.
        """
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        return ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extrae y normaliza el texto plano del árbol limpio.

        Args:
            soup: Árbol BeautifulSoup ya limpio.

        Returns:
            Texto con espacios normalizados y sin padding innecesario.
        """
        raw_text = soup.get_text(separator=" ", strip=True)
        return _WHITESPACE_RE.sub(" ", raw_text).strip()

    @staticmethod
    def _extract_category(url: str) -> tuple[str, str]:
        """Extrae categoría y subcategoría a partir del path de la URL.

        El path se parte por ``/``. El segmento ``personas`` se usa como
        prefijo; la categoría es el siguiente segmento, y la subcategoría
        el que le sigue (si existe).

        Examples::

            /personas/cuentas/ahorros → ("cuentas", "ahorros")
            /personas/creditos/consumo → ("creditos", "consumo")
            /personas                 → ("general", "")

        Args:
            url: URL absoluta de la página.

        Returns:
            Tupla ``(category, subcategory)``.
        """
        path = urlparse(url).path
        segments = [s for s in path.split("/") if s]

        if segments and segments[0] == "personas":
            segments = segments[1:]

        category = segments[0] if segments else "general"
        subcategory = segments[1] if len(segments) > 1 else ""

        return category, subcategory
