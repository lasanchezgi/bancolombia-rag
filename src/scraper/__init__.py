"""
Scraper package: crawlea y parsea páginas públicas de Bancolombia.

Expone Crawler para la descarga asíncrona de HTML, Parser para la extracción
de texto estructurado y ScraperStorage para la persistencia de documentos crudos.
"""

from .crawler import Crawler
from .parser import Parser
from .storage import ScraperStorage

__all__ = ["Crawler", "Parser", "ScraperStorage"]
