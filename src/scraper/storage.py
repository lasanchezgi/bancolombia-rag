"""
Persistencia de documentos crudos del scraper en disco como archivos JSON.

Cada documento se guarda como ``<slug>.json`` dentro del directorio configurado,
donde el slug se deriva de la URL de la página. Permite recargar todos los
documentos para el pipeline de limpieza y chunking posterior.
"""

from __future__ import annotations

from pathlib import Path


class ScraperStorage:
    """Guarda y carga documentos scrapeados desde el sistema de archivos.

    Attributes:
        output_dir: Directorio raíz donde se escriben los archivos JSON.
    """

    def __init__(self, output_dir: Path) -> None:
        """Inicializa el storage apuntando al directorio de salida.

        Args:
            output_dir: Ruta al directorio donde se guardan los JSON.
                        Se crea automáticamente si no existe.
        """
        self.output_dir = output_dir

    def save(self, document: dict[str, str]) -> Path:
        """Persiste un documento en disco como archivo JSON.

        El nombre de archivo se genera a partir de la URL del documento
        para evitar duplicados y facilitar la depuración.

        Args:
            document: Dict con al menos las claves ``url``, ``title``,
                      ``text`` y ``metadata``.

        Returns:
            Ruta absoluta al archivo JSON creado.

        Raises:
            KeyError: Si el dict no contiene la clave ``url``.
        """
        raise NotImplementedError

    def load_all(self) -> list[dict[str, str]]:
        """Carga todos los documentos JSON del directorio de salida.

        Returns:
            Lista de dicts de documentos. Lista vacía si el directorio
            no contiene archivos ``.json``.
        """
        raise NotImplementedError

    @staticmethod
    def _url_to_slug(url: str) -> str:
        """Convierte una URL en un slug seguro para usar como nombre de archivo.

        Args:
            url: URL absoluta de la página.

        Returns:
            String con caracteres seguros para nombres de archivo.
        """
        raise NotImplementedError
