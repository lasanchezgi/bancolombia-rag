"""
Script de verificación end-to-end del pipeline de trazabilidad RAG.

Ejecuta varias preguntas reales contra el agente y muestra las métricas
de trazabilidad capturadas: scores, tiempos, queries frecuentes y gaps.

Uso:
    uv run python scripts/test_tracing.py
"""

import json
import os

from src.agent.agent import RAGAgent

agent = RAGAgent(openai_api_key=os.environ["OPENAI_API_KEY"])

preguntas = [
    "¿Qué tarjetas de crédito tiene Bancolombia?",
    "¿Cuáles son los beneficios de la cuenta AFC?",
    "¿Cuál es el precio del dólar hoy?",  # out of scope
]

for pregunta in preguntas:
    print(f"\n{'='*60}")
    print(f"Pregunta: {pregunta}")
    respuesta = agent.ask(pregunta)
    print(f"Respuesta: {respuesta[:200]}...")

print("\n" + "=" * 60)
print("MÉTRICAS DE TRAZABILIDAD")
print("=" * 60)

stats = agent.logger.get_stats()
print("\nStats generales:")
print(json.dumps(stats, indent=2, ensure_ascii=False))

metrics = agent.logger.get_rag_performance_metrics()
print("\nRAG Performance:")
print(json.dumps(metrics, indent=2, ensure_ascii=False))

traces = agent.logger.get_mcp_traces(limit=10)
print(f"\nÚltimas llamadas MCP ({len(traces)} registros):")
for t in traces:
    print(f"  [{t['tool_name']}] query='{str(t.get('query_sent', ''))[:40]}'")
    if t.get("top_chromadb_score") is not None:
        print(f"    chromadb={t['top_chromadb_score']:.3f}", end="")
    else:
        print("    chromadb=N/A", end="")
    if t.get("top_rerank_score") is not None:
        print(f"  rerank={t['top_rerank_score']:.3f}")
    else:
        print("  rerank=N/A")
    print(f"    retrieval={t.get('retrieval_ms')}ms reranking={t.get('reranking_ms')}ms")

gaps = agent.logger.get_coverage_gaps()
print("\nGaps de cobertura detectados:")
for g in gaps:
    print(f"  '{str(g['question'])[:60]}' - {g['count']} veces")
