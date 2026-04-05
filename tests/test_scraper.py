"""
Tests unitarios para el paquete scraper.

Cubre Crawler (filtrado de URLs, fetch_page con mocks HTTP),
Parser (extracción de título/texto/categoría, casos borde) y
ScraperStorage (ciclo save/load, slugs, URLs ya scrapeadas).

No realiza peticiones de red reales: usa unittest.mock y respuestas HTTP
sintéticas para aislar cada unidad.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scraper.crawler import Crawler, _BLOCKED_PATTERNS
from src.scraper.parser import Parser
from src.scraper.storage import ScraperStorage

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_HTML = """
<html>
<head><title>Cuentas de Ahorro | Bancolombia</title></head>
<body>
  <header class="header-main">Menú principal</header>
  <nav>Navegación</nav>
  <script>alert('noise')</script>
  <style>.foo { color: red }</style>
  <h1>Cuentas de Ahorro</h1>
  <main>
    <p>Abre tu cuenta de ahorros fácilmente.</p>
    <p>Tasas competitivas y sin cuota de manejo.</p>
  </main>
  <footer>Pie de página</footer>
</body>
</html>
"""

SAMPLE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.bancolombia.com/personas/cuentas/ahorros</loc></url>
  <url><loc>https://www.bancolombia.com/personas/creditos/consumo</loc></url>
  <url><loc>https://www.bancolombia.com/personas/buscador</loc></url>
  <url><loc>https://www.bancolombia.com/personas/formulario/contacto</loc></url>
  <url><loc>https://www.bancolombia.com/personas/inversiones</loc></url>
</urlset>
"""


@pytest.fixture()
def crawler() -> Crawler:
    return Crawler(base_url="https://www.bancolombia.com", max_pages=10)


@pytest.fixture()
def parser() -> Parser:
    return Parser()


@pytest.fixture()
def storage(tmp_path: Path) -> ScraperStorage:
    return ScraperStorage(tmp_path / "raw")


@pytest.fixture()
def sample_raw_page() -> dict:
    return {"url": "https://www.bancolombia.com/personas/cuentas/ahorros", "html": SAMPLE_HTML}


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Crawler
# ──────────────────────────────────────────────────────────────────────────────


class TestCrawlerInit:
    def test_stores_base_url(self, crawler: Crawler) -> None:
        """Crawler debe almacenar el base_url recibido."""
        assert crawler.base_url == "https://www.bancolombia.com"

    def test_stores_max_pages(self, crawler: Crawler) -> None:
        """Crawler debe almacenar el max_pages recibido."""
        assert crawler.max_pages == 10


class TestCrawlerIsBlocked:
    def test_blocked_url_returns_true(self, crawler: Crawler) -> None:
        """Una URL con patrón bloqueado debe ser filtrada."""
        assert crawler._is_blocked("https://www.bancolombia.com/personas/buscador") is True

    def test_clean_url_returns_false(self, crawler: Crawler) -> None:
        """Una URL sin patrones bloqueados no debe ser filtrada."""
        assert crawler._is_blocked("https://www.bancolombia.com/personas/cuentas/ahorros") is False

    @pytest.mark.parametrize("pattern", _BLOCKED_PATTERNS)
    def test_all_blocked_patterns_are_filtered(self, crawler: Crawler, pattern: str) -> None:
        """Cada patrón bloqueado debe ser detectado correctamente."""
        url = f"https://www.bancolombia.com/{pattern}/page"
        assert crawler._is_blocked(url) is True


class TestFetchSitemapUrls:
    def test_returns_only_allowed_urls(self, crawler: Crawler) -> None:
        """fetch_sitemap_urls debe excluir URLs con patrones bloqueados."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_SITEMAP
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            urls = crawler.fetch_sitemap_urls("https://www.bancolombia.com/sitemap-personas.xml")

        # buscador y formulario deben haber sido filtrados
        assert all("buscador" not in u and "formulario" not in u for u in urls)

    def test_returns_list_type(self, crawler: Crawler) -> None:
        """fetch_sitemap_urls debe devolver una lista."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_SITEMAP
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            result = crawler.fetch_sitemap_urls("https://x.com/sitemap.xml")

        assert isinstance(result, list)

    def test_correct_url_count_after_filtering(self, crawler: Crawler) -> None:
        """Deben quedar 3 URLs permitidas de las 5 del sitemap de muestra."""
        mock_response = MagicMock()
        mock_response.text = SAMPLE_SITEMAP
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            urls = crawler.fetch_sitemap_urls("https://www.bancolombia.com/sitemap-personas.xml")

        # cuentas/ahorros, creditos/consumo, inversiones → 3 permitidas
        assert len(urls) == 3


class TestFetchPage:
    @pytest.mark.asyncio
    async def test_returns_dict_on_success(self, crawler: Crawler) -> None:
        """fetch_page debe devolver un dict con url, html y status_code."""
        mock_response = AsyncMock()
        mock_response.text = SAMPLE_HTML
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await crawler.fetch_page("https://www.bancolombia.com/personas/cuentas", mock_client)

        assert result is not None
        assert result["url"] == "https://www.bancolombia.com/personas/cuentas"
        assert result["status_code"] == 200
        assert "html" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self, crawler: Crawler) -> None:
        """fetch_page debe devolver None ante un error HTTP."""
        import httpx

        mock_client = AsyncMock()
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("404", request=mock_request, response=mock_response)
        )

        result = await crawler.fetch_page("https://www.bancolombia.com/not-found", mock_client)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, crawler: Crawler) -> None:
        """fetch_page debe devolver None ante un timeout."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        result = await crawler.fetch_page("https://www.bancolombia.com/slow", mock_client)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Parser
# ──────────────────────────────────────────────────────────────────────────────


class TestParserParse:
    def test_returns_required_keys(self, parser: Parser, sample_raw_page: dict) -> None:
        """El resultado debe contener todas las claves canónicas."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        for key in ("url", "title", "text", "category", "subcategory", "extraction_date", "char_count"):
            assert key in result

    def test_title_from_h1(self, parser: Parser, sample_raw_page: dict) -> None:
        """El título debe tomarse del primer <h1> cuando existe."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert result["title"] == "Cuentas de Ahorro"

    def test_script_content_excluded(self, parser: Parser, sample_raw_page: dict) -> None:
        """El contenido de <script> no debe aparecer en el texto."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert "alert" not in result["text"]

    def test_style_content_excluded(self, parser: Parser, sample_raw_page: dict) -> None:
        """El contenido de <style> no debe aparecer en el texto."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert ".foo" not in result["text"]

    def test_nav_content_excluded(self, parser: Parser, sample_raw_page: dict) -> None:
        """El contenido de <nav> no debe aparecer en el texto."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert "Navegación" not in result["text"]

    def test_category_extracted_from_url(self, parser: Parser, sample_raw_page: dict) -> None:
        """La categoría debe ser el segundo segmento del path."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert result["category"] == "cuentas"

    def test_subcategory_extracted_from_url(self, parser: Parser, sample_raw_page: dict) -> None:
        """La subcategoría debe ser el tercer segmento del path."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert result["subcategory"] == "ahorros"

    def test_char_count_matches_text_length(self, parser: Parser, sample_raw_page: dict) -> None:
        """char_count debe coincidir con len(text)."""
        result = parser.parse(sample_raw_page)
        assert result is not None
        assert result["char_count"] == len(result["text"])

    def test_returns_none_for_empty_body(self, parser: Parser) -> None:
        """parse() debe devolver None si el texto extraído queda vacío."""
        raw = {
            "url": "https://www.bancolombia.com/personas",
            "html": "<html><body><script>code</script></body></html>",
        }
        assert parser.parse(raw) is None


class TestParserExtractCategory:
    @pytest.mark.parametrize(
        "url,expected_category,expected_sub",
        [
            ("https://www.bancolombia.com/personas/cuentas/ahorros", "cuentas", "ahorros"),
            ("https://www.bancolombia.com/personas/creditos/consumo", "creditos", "consumo"),
            ("https://www.bancolombia.com/personas", "general", ""),
            ("https://www.bancolombia.com/personas/inversiones", "inversiones", ""),
        ],
    )
    def test_category_extraction(self, url: str, expected_category: str, expected_sub: str) -> None:
        """_extract_category debe parsear correctamente distintos paths."""
        cat, sub = Parser._extract_category(url)
        assert cat == expected_category
        assert sub == expected_sub


# ──────────────────────────────────────────────────────────────────────────────
# Tests: ScraperStorage
# ──────────────────────────────────────────────────────────────────────────────


class TestScraperStorage:
    def test_output_dir_created_on_init(self, tmp_path: Path) -> None:
        """__init__ debe crear output_dir si no existe."""
        target = tmp_path / "new_dir" / "raw"
        assert not target.exists()
        ScraperStorage(target)
        assert target.exists()

    def test_save_creates_json_file(self, storage: ScraperStorage) -> None:
        """save() debe crear un archivo .json en output_dir."""
        doc = {"url": "https://www.bancolombia.com/personas/cuentas", "title": "Cuentas", "text": "Texto"}
        path = storage.save(doc)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_filename_derived_from_url(self, storage: ScraperStorage) -> None:
        """El nombre del archivo debe derivarse de la URL según la convención."""
        doc = {"url": "https://www.bancolombia.com/personas/cuentas/ahorros", "text": "X"}
        path = storage.save(doc)
        assert "personas" in path.name
        assert "cuentas" in path.name

    def test_save_roundtrip(self, storage: ScraperStorage) -> None:
        """Un documento guardado con save() debe recuperarse íntegro con load_all()."""
        doc = {"url": "https://www.bancolombia.com/personas/inversiones", "title": "Inversiones", "text": "Texto largo"}
        storage.save(doc)
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0]["url"] == doc["url"]
        assert loaded[0]["title"] == doc["title"]

    def test_load_all_returns_empty_list_when_no_files(self, storage: ScraperStorage) -> None:
        """load_all() debe devolver [] si no hay archivos JSON."""
        assert storage.load_all() == []

    def test_load_all_skips_invalid_json(self, storage: ScraperStorage) -> None:
        """load_all() no debe romper si hay un archivo JSON corrupto."""
        bad_file = storage.output_dir / "corrupt.json"
        bad_file.write_text("{ not valid json", encoding="utf-8")
        result = storage.load_all()
        assert isinstance(result, list)
        assert len(result) == 0  # el archivo corrupto se ignora

    def test_get_urls_already_scraped_returns_set(self, storage: ScraperStorage) -> None:
        """get_urls_already_scraped() debe devolver un set."""
        assert isinstance(storage.get_urls_already_scraped(), set)

    def test_get_urls_already_scraped_contains_saved_url(self, storage: ScraperStorage) -> None:
        """La URL de un documento guardado debe aparecer en el set."""
        url = "https://www.bancolombia.com/personas/cuentas"
        storage.save({"url": url, "text": "contenido"})
        scraped = storage.get_urls_already_scraped()
        assert url in scraped

    def test_save_content_is_valid_json(self, storage: ScraperStorage) -> None:
        """El archivo escrito por save() debe ser JSON válido."""
        doc = {"url": "https://www.bancolombia.com/personas/seguros", "text": "Texto con ñ y tildes á"}
        path = storage.save(doc)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["url"] == doc["url"]
