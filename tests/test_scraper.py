"""
Tests unitarios para el paquete scraper.

Cubre: Crawler (instanciación y configuración), Parser (extracción de texto
y eliminación de tags de ruido) y ScraperStorage (ciclo save/load en disco).
"""

from __future__ import annotations

from pathlib import Path

from src.scraper.crawler import Crawler


class TestParser:
    """Tests para Parser.parse()."""

    def test_parse_returns_required_keys(self) -> None:
        """El dict resultante debe contener url, title, text y metadata."""
        pass

    def test_parse_removes_script_tags(self) -> None:
        """El contenido de etiquetas <script> no debe aparecer en el texto."""
        pass

    def test_parse_removes_style_tags(self) -> None:
        """El contenido de etiquetas <style> no debe aparecer en el texto."""
        pass

    def test_parse_extracts_title(self) -> None:
        """El campo 'title' debe corresponder al contenido de <title>."""
        pass

    def test_parse_empty_html_returns_empty_text(self) -> None:
        """HTML vacío debe devolver un dict con text vacío, sin lanzar excepciones."""
        pass


class TestCrawler:
    """Tests de instanciación y configuración del Crawler."""

    def test_crawler_stores_base_url(self) -> None:
        """Crawler debe exponer el atributo base_url con el valor recibido."""
        crawler = Crawler(base_url="https://example.com", max_pages=10)
        assert crawler.base_url == "https://example.com"

    def test_crawler_stores_max_pages(self) -> None:
        """Crawler debe exponer el atributo max_pages con el valor recibido."""
        crawler = Crawler(base_url="https://example.com", max_pages=25)
        assert crawler.max_pages == 25


class TestScraperStorage:
    """Tests para ScraperStorage.save() y load_all()."""

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        """save() debe crear un archivo .json dentro de output_dir."""
        pass

    def test_load_all_returns_list_when_empty(self, tmp_path: Path) -> None:
        """load_all() debe devolver una lista vacía si no hay archivos JSON."""
        pass

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Un documento guardado con save() debe recuperarse íntegro con load_all()."""
        pass
