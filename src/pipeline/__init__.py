"""
Pipeline package: limpia y divide texto crudo en chunks para embedding.

Expone Cleaner para normalización de texto y Chunker para la estrategia
de ventana deslizante con solapamiento configurable.
"""

from .chunker import Chunker
from .cleaner import Cleaner

__all__ = ["Cleaner", "Chunker"]
