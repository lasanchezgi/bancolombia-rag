"""
Embedder que genera vectores de texto usando la API de OpenAI.

Usa el modelo ``text-embedding-3-small`` por defecto. Soporta llamadas en
batch para aprovechar los límites de la API y minimizar la latencia total
al procesar grandes volúmenes de chunks durante el pipeline de indexación.
"""

from __future__ import annotations

from openai import OpenAI


class Embedder:
    """Genera embeddings para listas de textos usando la API de OpenAI.

    Attributes:
        model: Identificador del modelo de embeddings de OpenAI.
        client: Instancia del cliente OpenAI.
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Inicializa el Embedder y el cliente de OpenAI.

        La clave de API se toma automáticamente de la variable de entorno
        ``OPENAI_API_KEY``.

        Args:
            model: Identificador del modelo de embeddings a usar.
        """
        self.model = model
        self.client = OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos en batch.

        Args:
            texts: Lista de strings a convertir en vectores.

        Returns:
            Lista de vectores de embedding, uno por cada texto de entrada.
            El orden se preserva (``result[i]`` corresponde a ``texts[i]``).

        Raises:
            openai.OpenAIError: Si la llamada a la API falla.
        """
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        """Genera el embedding de un único texto.

        Wrapper de conveniencia sobre :meth:`embed` para casos donde
        solo se necesita vectorizar un texto (e.g. una query de usuario).

        Args:
            text: Texto a convertir en vector.

        Returns:
            Vector de embedding como lista de floats.
        """
        raise NotImplementedError
