"""
Limpiador de texto que normaliza el contenido crudo extraído por el scraper.

Aplica regex para eliminar URLs, caracteres especiales no deseados y
espacios/saltos de línea redundantes, devolviendo un dict enriquecido
con los campos "clean_text" y "word_count" listos para el Chunker.
"""

from __future__ import annotations

import re
from typing import Any

_URL_RE = re.compile(r"https?://\S+")
# Conserva: letras (incluyendo unicode/español), dígitos, espacios, puntuación básica
_ALLOWED_CHARS_RE = re.compile(r"[^\w\s.,;:¿?¡!()\-%]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_MIN_TEXT_LENGTH = 50


class Cleaner:
    """Normaliza texto crudo proveniente del scraper.

    Aplica las siguientes transformaciones en orden sobre el campo ``"text"``
    del dict de entrada:

    1. Eliminar URLs (``https?://...``).
    2. Eliminar caracteres especiales fuera del conjunto permitido.
    3. Colapsar múltiples espacios/tabs en uno solo.
    4. Colapsar más de 2 saltos de línea consecutivos en exactamente 2.
    5. Strip del resultado.
    6. Descartar si el texto resultante tiene menos de 50 caracteres.
    """

    def clean(self, page: dict[str, Any]) -> dict[str, Any] | None:
        """Limpia el texto de una página cruda del scraper.

        Args:
            page: Dict del scraper con al menos la clave ``"text"``.

        Returns:
            Dict con los mismos campos originales más ``"clean_text"``
            (texto limpio) y ``"word_count"`` (palabras en clean_text).
            Devuelve ``None`` si el texto limpio tiene menos de 50 caracteres.
        """
        text: str = page["text"]

        # 1. Eliminar URLs
        text = _URL_RE.sub("", text)
        # 2. Eliminar caracteres especiales no permitidos
        text = _ALLOWED_CHARS_RE.sub("", text)
        # 3. Colapsar espacios y tabs múltiples
        text = _MULTI_SPACE_RE.sub(" ", text)
        # 4. Colapsar 3+ saltos de línea a exactamente 2
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        # 5. Strip
        text = text.strip()

        if len(text) < _MIN_TEXT_LENGTH:
            return None

        return {**page, "clean_text": text, "word_count": len(text.split())}
