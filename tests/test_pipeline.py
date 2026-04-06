"""
Tests unitarios para los módulos Cleaner y Chunker del pipeline.

Cubre limpieza de texto (URLs, caracteres especiales, espacios, newlines,
longitud mínima) y chunking (claves requeridas, texto corto, IDs únicos,
propagación de metadatos, edge cases).

No realiza peticiones de red ni accesos a disco reales.
"""

from __future__ import annotations

import pytest

from src.pipeline.cleaner import Cleaner
from src.pipeline.chunker import Chunker

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures y datos de prueba
# ──────────────────────────────────────────────────────────────────────────────

LONG_TEXT = (
    "Las cuentas de ahorro de Bancolombia ofrecen tasas competitivas. "
    "Puedes abrir tu cuenta de manera facil y rapida sin cuota de manejo. "
    "Los beneficios incluyen transferencias gratuitas, acceso a cajeros, "
    "y seguimiento digital en tiempo real de tus movimientos financieros. "
    "Ademas contamos con atencion al cliente disponible 24 horas al dia, "
    "los 7 dias de la semana para resolver todas tus dudas y consultas."
)

SAMPLE_PAGE = {
    "url": "https://www.bancolombia.com/personas/cuentas/ahorros",
    "title": "Cuentas de Ahorro",
    "text": LONG_TEXT,
    "category": "cuentas",
    "subcategory": "ahorros",
    "extraction_date": "2024-01-01T00:00:00",
    "char_count": len(LONG_TEXT),
}


@pytest.fixture()
def cleaner() -> Cleaner:
    return Cleaner()


@pytest.fixture()
def chunker() -> Chunker:
    return Chunker()


@pytest.fixture()
def cleaned_page(cleaner: Cleaner) -> dict:
    result = cleaner.clean(SAMPLE_PAGE)
    assert result is not None
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Cleaner
# ──────────────────────────────────────────────────────────────────────────────


class TestCleaner:
    def test_clean_removes_urls(self, cleaner: Cleaner) -> None:
        """URLs deben eliminarse del texto limpio."""
        page = {**SAMPLE_PAGE, "text": LONG_TEXT + " Visita https://www.bancolombia.com/info para mas datos sobre servicios."}
        result = cleaner.clean(page)
        assert result is not None
        assert "https://" not in result["clean_text"]
        assert "http://" not in result["clean_text"]

    def test_clean_collapses_multiple_spaces(self, cleaner: Cleaner) -> None:
        """Multiples espacios consecutivos deben colapsar en uno solo."""
        page = {**SAMPLE_PAGE, "text": "Texto   con   muchos   espacios   redundantes entre palabras comunes de uso diario."}
        result = cleaner.clean(page)
        if result is not None:
            assert "  " not in result["clean_text"]

    def test_clean_collapses_multiple_newlines(self, cleaner: Cleaner) -> None:
        """Tres o mas saltos de linea consecutivos deben reducirse a dos."""
        page = {**SAMPLE_PAGE, "text": "Parrafo uno con bastante contenido util.\n\n\n\nParrafo dos con suficiente contenido para superar el minimo requerido."}
        result = cleaner.clean(page)
        if result is not None:
            assert "\n\n\n" not in result["clean_text"]

    def test_clean_strips_leading_trailing_whitespace(self, cleaner: Cleaner) -> None:
        """El texto limpio no debe tener espacios al inicio ni al final."""
        page = {**SAMPLE_PAGE, "text": "   " + LONG_TEXT + "   "}
        result = cleaner.clean(page)
        assert result is not None
        assert result["clean_text"] == result["clean_text"].strip()

    def test_clean_returns_none_for_short_text(self, cleaner: Cleaner) -> None:
        """Texto de menos de 50 caracteres tras limpieza debe devolver None."""
        page = {**SAMPLE_PAGE, "text": "Texto corto."}
        assert cleaner.clean(page) is None

    def test_clean_adds_clean_text_field(self, cleaner: Cleaner) -> None:
        """El resultado debe contener el campo 'clean_text'."""
        result = cleaner.clean(SAMPLE_PAGE)
        assert result is not None
        assert "clean_text" in result

    def test_clean_adds_word_count_field(self, cleaner: Cleaner) -> None:
        """El resultado debe contener el campo 'word_count'."""
        result = cleaner.clean(SAMPLE_PAGE)
        assert result is not None
        assert "word_count" in result

    def test_clean_word_count_matches_text(self, cleaner: Cleaner) -> None:
        """word_count debe coincidir con len(clean_text.split())."""
        result = cleaner.clean(SAMPLE_PAGE)
        assert result is not None
        assert result["word_count"] == len(result["clean_text"].split())

    def test_clean_preserves_original_fields(self, cleaner: Cleaner) -> None:
        """Todos los campos originales del dict deben conservarse en el resultado."""
        result = cleaner.clean(SAMPLE_PAGE)
        assert result is not None
        for key in ("url", "title", "category", "subcategory", "extraction_date"):
            assert key in result
            assert result[key] == SAMPLE_PAGE[key]

    def test_clean_removes_special_characters(self, cleaner: Cleaner) -> None:
        """Caracteres como @, #, $ deben eliminarse del texto limpio."""
        page = {**SAMPLE_PAGE, "text": LONG_TEXT + " @usuario #hashtag $precio caret ampersand caracteres raros especiales."}
        result = cleaner.clean(page)
        assert result is not None
        assert "@" not in result["clean_text"]
        assert "#" not in result["clean_text"]
        assert "$" not in result["clean_text"]


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Chunker
# ──────────────────────────────────────────────────────────────────────────────


class TestChunker:
    def test_chunk_returns_list_of_dicts(self, chunker: Chunker, cleaned_page: dict) -> None:
        """chunk() debe devolver una lista de dicts."""
        result = chunker.chunk(cleaned_page)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_chunk_has_required_keys(self, chunker: Chunker, cleaned_page: dict) -> None:
        """Cada chunk debe contener todas las claves requeridas."""
        result = chunker.chunk(cleaned_page)
        assert len(result) > 0
        required = {
            "chunk_id", "url", "title", "category", "subcategory",
            "extraction_date", "chunk_index", "total_chunks", "text", "word_count",
        }
        for chunk in result:
            assert required.issubset(chunk.keys())

    def test_chunk_empty_text_returns_empty_list(self, chunker: Chunker, cleaned_page: dict) -> None:
        """Texto con menos de 10 palabras debe retornar lista vacia."""
        short_page = {**cleaned_page, "clean_text": "Hola mundo."}
        assert chunker.chunk(short_page) == []

    def test_chunk_single_chunk_for_short_text(self, chunker: Chunker, cleaned_page: dict) -> None:
        """Texto de menos de 500 palabras debe generar exactamente 1 chunk."""
        # LONG_TEXT tiene ~80 palabras
        result = chunker.chunk(cleaned_page)
        assert len(result) == 1

    def test_chunk_ids_are_unique(self, chunker: Chunker, cleaned_page: dict) -> None:
        """Todos los chunk_id del resultado deben ser unicos."""
        result = chunker.chunk(cleaned_page)
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_copied_to_all_chunks(self, chunker: Chunker, cleaned_page: dict) -> None:
        """url, title y category deben propagarse a cada chunk."""
        result = chunker.chunk(cleaned_page)
        for chunk in result:
            assert chunk["url"] == cleaned_page["url"]
            assert chunk["title"] == cleaned_page["title"]
            assert chunk["category"] == cleaned_page["category"]

    def test_chunk_index_and_total_consistent(self, chunker: Chunker, cleaned_page: dict) -> None:
        """chunk_index debe ser secuencial y total_chunks correcto."""
        result = chunker.chunk(cleaned_page)
        total = result[0]["total_chunks"]
        assert total == len(result)
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == total

    def test_chunk_word_count_matches_text(self, chunker: Chunker, cleaned_page: dict) -> None:
        """word_count de cada chunk debe coincidir con len(text.split())."""
        result = chunker.chunk(cleaned_page)
        for chunk in result:
            assert chunk["word_count"] == len(chunk["text"].split())

    def test_chunk_large_text_produces_multiple_chunks(self, chunker: Chunker, cleaned_page: dict) -> None:
        """Texto de mas de 500 palabras debe producir mas de 1 chunk."""
        words = ["palabra"] * 600
        large_page = {**cleaned_page, "clean_text": " ".join(words)}
        result = chunker.chunk(large_page)
        assert len(result) > 1
