"""
Persistencia de documentos procesados del scraper en disco como archivos JSON.

Cada documento se guarda como ``<slug>.json`` dentro del directorio configurado,
donde el slug se deriva de la URL de la página. Permite recargar todos los
documentos para el pipeline de limpieza y chunking posterior, y detectar
URLs ya scrapeadas para poder reanudar ejecuciones interrumpidas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScraperStorage:
    """Guarda y carga documentos scrapeados desde el sistema de archivos.

    Attributes:
        output_dir: Directorio raíz donde se escriben los archivos JSON.
    """

    def __init__(self, output_dir: Path) -> None:
        """Inicializa el storage y crea el directorio de salida si no existe.

        Args:
            output_dir: Ruta al directorio donde se guardan los JSON.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, page: dict[str, Any]) -> Path:
        """Persiste un documento en disco como archivo JSON.

        El nombre de archivo se genera a partir de la URL del documento
        según la convención:
        ``url.replace("https://www.bancolombia.com/", "").replace("/", "_").replace("-", "_") + ".json"``

        Args:
            page: Dict con al menos la clave ``"url"`` y el resto de campos
                  del documento parseado.

        Returns:
            Ruta absoluta al archivo JSON creado.

        Raises:
            KeyError: Si el dict no contiene la clave ``"url"``.
        """
        filename = self._url_to_filename(page["url"])
        path = self.output_dir / filename
        path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_all(self) -> list[dict[str, Any]]:
        """Carga todos los documentos JSON del directorio de salida.

        Los archivos que fallen al parsear se ignoran y se loguea un warning.

        Returns:
            Lista de dicts. Lista vacía si el directorio no contiene ``.json``.
        """
        documents: list[dict[str, Any]] = []
        for path in sorted(self.output_dir.glob("*.json")):
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Error leyendo %s: %s", path.name, exc)
        return documents

    def get_urls_already_scraped(self) -> set[str]:
        """Devuelve el conjunto de URLs ya persistidas en disco.

        Permite reanudar una ejecución interrumpida sin re-scrapear páginas
        que ya fueron procesadas y guardadas.

        Returns:
            Set de strings con las URLs de los documentos ya guardados.
        """
        urls: set[str] = set()
        for doc in self.load_all():
            url = doc.get("url")
            if url:
                urls.add(url)
        return urls

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Convierte una URL en un nombre de archivo ``.json`` seguro.

        Elimina el prefijo ``https://www.bancolombia.com/``, luego reemplaza
        ``/`` y ``-`` por ``_``.

        Args:
            url: URL absoluta de la página.

        Returns:
            Nombre de archivo con extensión ``.json``.
        """
        slug = url.replace("https://www.bancolombia.com/", "")
        slug = slug.replace("/", "_").replace("-", "_")
        slug = slug.strip("_") or "index"
        return f"{slug}.json"
