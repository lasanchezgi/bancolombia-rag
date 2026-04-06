"""
Chunker que divide texto limpio en fragmentos con solapamiento.

Implementa una estrategia RecursiveCharacterTextSplitter manual usando
conteo de palabras como proxy de tokens. Intenta mantener la coherencia
semántica dividiendo primero por párrafos, luego oraciones y finalmente
espacios simples.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

_CHUNK_SIZE = 500  # palabras por chunk
_CHUNK_OVERLAP = 50  # palabras compartidas entre chunks adyacentes
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_MIN_WORDS = 10


class Chunker:
    """Divide texto limpio en chunks con solapamiento configurable.

    Usa conteo de palabras como proxy de tokens. Los separadores se
    aplican en orden de preferencia semántica: párrafo > línea > oración
    > espacio > carácter.
    """

    def chunk(self, page: dict[str, Any]) -> list[dict[str, Any]]:
        """Divide el texto limpio de una página en chunks con metadatos.

        Args:
            page: Dict limpio con al menos ``"clean_text"``, ``"url"``,
                  ``"title"``, ``"category"``, ``"subcategory"`` y
                  ``"extraction_date"``.

        Returns:
            Lista de dicts, uno por chunk, con las claves:
            ``chunk_id``, ``url``, ``title``, ``category``,
            ``subcategory``, ``extraction_date``, ``chunk_index``,
            ``total_chunks``, ``text``, ``word_count``.
            Lista vacía si el texto tiene menos de 10 palabras.
        """
        text: str = page["clean_text"]
        url: str = page["url"]

        words = text.split()
        if len(words) < _MIN_WORDS:
            return []

        # Divide recursivamente hasta obtener piezas <= _CHUNK_SIZE palabras
        pieces = self._split_recursive(text, _SEPARATORS)
        # Fusiona las piezas en ventanas de _CHUNK_SIZE con solapamiento
        raw_chunks = self._merge_pieces(pieces)

        url_slug = self._url_to_slug(url)
        total = len(raw_chunks)

        return [
            {
                "chunk_id": f"{url_slug}_{i}",
                "url": url,
                "title": page.get("title", ""),
                "category": page.get("category", ""),
                "subcategory": page.get("subcategory", ""),
                "extraction_date": page.get("extraction_date", ""),
                "chunk_index": i,
                "total_chunks": total,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
            }
            for i, chunk_text in enumerate(raw_chunks)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────────

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Divide text en piezas de <= _CHUNK_SIZE palabras, probando separadores en orden.

        Args:
            text: Texto a dividir.
            separators: Lista de separadores a probar en orden de preferencia.

        Returns:
            Lista de strings donde cada uno tiene <= _CHUNK_SIZE palabras.
        """
        if not text.strip():
            return []

        if len(text.split()) <= _CHUNK_SIZE:
            return [text]

        if not separators:
            # Último recurso: dividir por palabras directamente
            words = text.split()
            return [" ".join(words[i : i + _CHUNK_SIZE]) for i in range(0, len(words), _CHUNK_SIZE)]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            words = text.split()
            return [" ".join(words[i : i + _CHUNK_SIZE]) for i in range(0, len(words), _CHUNK_SIZE)]

        pieces = text.split(sep)
        result: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece.split()) <= _CHUNK_SIZE:
                result.append(piece)
            else:
                result.extend(self._split_recursive(piece, rest))

        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Fusiona piezas pequeñas en chunks de ~_CHUNK_SIZE palabras con solapamiento.

        Args:
            pieces: Lista de strings pequeños obtenidos de _split_recursive.

        Returns:
            Lista de chunks con solapamiento de _CHUNK_OVERLAP palabras.
        """
        chunks: list[str] = []
        current_words: list[str] = []

        for piece in pieces:
            piece_words = piece.split()
            if not piece_words:
                continue

            if len(current_words) + len(piece_words) <= _CHUNK_SIZE:
                current_words.extend(piece_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                # Nuevo chunk comienza con las últimas _CHUNK_OVERLAP palabras del anterior
                overlap = current_words[-_CHUNK_OVERLAP:] if current_words else []
                current_words = overlap + piece_words

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    @staticmethod
    def _url_to_slug(url: str) -> str:
        """Convierte una URL en un slug seguro para usar como prefijo de chunk_id.

        Args:
            url: URL absoluta de la página.

        Returns:
            Path de la URL con ``/`` reemplazados por ``_``, sin dominio.
        """
        path = urlparse(url).path
        slug = path.strip("/").replace("/", "_").replace("-", "_")
        return slug or "index"
