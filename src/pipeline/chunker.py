"""
Chunker que divide texto limpio en fragmentos con solapamiento.

Implementa una estrategia de ventana deslizante basada en caracteres:
cada chunk tiene hasta CHUNK_SIZE caracteres y se solapa con el chunk
anterior en CHUNK_OVERLAP caracteres para preservar contexto en los bordes.
"""

from __future__ import annotations


class Chunker:
    """Divide texto en chunks de tamaño fijo con solapamiento configurable.

    Attributes:
        chunk_size: Tamaño máximo de cada chunk en caracteres.
        chunk_overlap: Número de caracteres compartidos entre chunks adyacentes.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Inicializa el Chunker con los parámetros de ventana deslizante.

        Args:
            chunk_size: Tamaño máximo de cada fragmento en caracteres.
            chunk_overlap: Solapamiento en caracteres entre chunks consecutivos.

        Raises:
            ValueError: Si chunk_overlap >= chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Divide el texto en chunks con ventana deslizante.

        Args:
            text: Texto limpio a fragmentar.

        Returns:
            Lista de strings donde cada elemento es un chunk. Devuelve una
            lista vacía si el texto de entrada está vacío.
        """
        raise NotImplementedError
