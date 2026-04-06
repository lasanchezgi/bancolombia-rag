"""
Script de prueba CLI para el RAGAgent de Bancolombia.

Instancia el agente, hace 3 preguntas de prueba e imprime las respuestas
junto con el tiempo de respuesta de cada una.

Uso:
    uv run python scripts/test_agent.py
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import RAGAgent  # noqa: E402

QUESTIONS = [
    "¿Qué cuentas de ahorro tiene Bancolombia?",
    "¿Cuáles son los beneficios de la tarjeta débito Black?",
    "¿Cómo puedo hacer giros internacionales?",
]


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY no está configurada en el entorno.")
        return

    print("Inicializando agente...")
    agent = RAGAgent(openai_api_key=api_key)
    print("Agente listo.\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"{'=' * 65}")
        print(f"[{i}/{len(QUESTIONS)}] Pregunta: {question}")
        print("-" * 65)

        t0 = time.perf_counter()
        answer = agent.ask(question)
        elapsed = time.perf_counter() - t0

        print(f"Respuesta:\n{answer}")
        print(f"\n⏱  Tiempo: {elapsed:.2f}s")
        print()

    print("=" * 65)
    print("Prueba completada.")


if __name__ == "__main__":
    main()
