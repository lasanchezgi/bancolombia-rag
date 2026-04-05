"""
Tests unitarios para el paquete pipeline.

Cubre: Cleaner (normalización de texto, casos borde) y Chunker
(ventana deslizante, solapamiento, inputs vacíos).
"""

from __future__ import annotations

import pytest

from src.pipeline.chunker import Chunker


class TestCleaner:
    """Tests para Cleaner.clean()."""

    def test_clean_collapses_multiple_spaces(self) -> None:
        """Espacios múltiples consecutivos deben colapsarse a uno solo."""
        pass

    def test_clean_collapses_multiple_newlines(self) -> None:
        """Saltos de línea múltiples deben reducirse a uno solo."""
        pass

    def test_clean_empty_string_returns_empty(self) -> None:
        """clean('') debe devolver '' sin lanzar excepciones."""
        pass

    def test_clean_strips_leading_trailing_whitespace(self) -> None:
        """El texto resultante no debe tener espacios al inicio ni al final."""
        pass


class TestChunker:
    """Tests para Chunker.chunk()."""

    def test_chunk_returns_nonempty_list_for_nonempty_input(self) -> None:
        """chunk() debe devolver al menos un chunk para texto no vacío."""
        pass

    def test_chunk_empty_string_returns_empty_list(self) -> None:
        """chunk('') debe devolver []."""
        pass

    def test_chunk_size_respected(self) -> None:
        """Ningún chunk debe superar chunk_size caracteres."""
        pass

    def test_chunk_overlap_respected(self) -> None:
        """Chunks adyacentes deben compartir chunk_overlap caracteres."""
        pass

    def test_chunk_overlap_greater_than_size_raises(self) -> None:
        """Crear Chunker con chunk_overlap >= chunk_size debe lanzar ValueError."""
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)
