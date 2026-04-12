"""
Compara la calidad del retrieval con y sin reranking para las mismas queries.

Invoca directamente los helpers del MCP server para ver el retrieval puro,
sin pasar por el agente completo.

Nota: search_knowledge_base está definida como función local dentro de
register_tools (no importable), por lo que se reutilizan los helpers del módulo
(_get_embedder, _get_repository, get_reranker) con la misma lógica.
"""

from __future__ import annotations

import time
from typing import Any

from src.mcp_server.tools import _get_embedder, _get_repository, get_reranker

QUERIES = [
    "¿Qué tarjetas de crédito tiene Bancolombia?",
    "¿Cuáles son los beneficios de la cuenta AFC?",
    "¿Cómo puedo hacer giros internacionales?",
    "¿Qué es la tarjeta débito Black?",
    "¿Qué cuentas tienen para pensionados?",
]

TOP_K = 3
N_CANDIDATES = TOP_K * 3  # candidatos extra para reranking


def search(query: str, use_reranking: bool) -> tuple[list[dict[str, Any]], float]:
    """Retrieval puro, equivalente a search_knowledge_base sin manejo de errores."""
    embedder = _get_embedder()
    repository = _get_repository()

    t0 = time.time()

    query_embedding = embedder.embed_texts([query])[0]
    n_candidates = N_CANDIDATES if use_reranking else TOP_K
    raw_results = repository.query(
        query_embedding=query_embedding,
        top_k=min(n_candidates, 30),
    )

    if use_reranking:
        results = get_reranker().rerank(query, raw_results, TOP_K)
    else:
        results = raw_results[:TOP_K]
        for r in results:
            r["rerank_score"] = None

    elapsed = time.time() - t0
    return results, elapsed


def truncate(url: str, max_len: int = 60) -> str:
    return url if len(url) <= max_len else url[:max_len - 3] + "..."


def print_results(label: str, results: list[dict], elapsed: float) -> None:
    print(f"\n  [{label}]  ({elapsed:.2f}s)")
    for i, r in enumerate(results, 1):
        chroma_score = r.get("score", r.get("relevance_score", "?"))
        rerank_score = r.get("rerank_score")
        score_str = f"chroma={chroma_score:.4f}"
        if rerank_score is not None:
            score_str += f"  rerank={rerank_score:.4f}"
        print(f"    {i}. {r['url']}")
        print(f"       {score_str}")


def main() -> None:
    # Almacena top-1 URLs para la tabla comparativa
    summary: list[dict] = []

    for query in QUERIES:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        results_no_rerank, t_no = search(query, use_reranking=False)
        print_results("Sin reranking", results_no_rerank, t_no)

        results_rerank, t_re = search(query, use_reranking=True)
        print_results("Con reranking", results_rerank, t_re)

        top1_no = results_no_rerank[0]["url"] if results_no_rerank else "—"
        top1_re = results_rerank[0]["url"] if results_rerank else "—"
        changed = "SI" if top1_no != top1_re else "no"

        summary.append(
            {
                "query": query,
                "sin_reranking": top1_no,
                "con_reranking": top1_re,
                "cambio": changed,
                "t_sin": t_no,
                "t_con": t_re,
            }
        )

    # Tabla comparativa final
    print("\n\n" + "=" * 70)
    print("TABLA COMPARATIVA — top-1 URL por método")
    print("=" * 70)

    col_q = 40
    col_url = 52
    header = (
        f"{'Query':<{col_q}} | "
        f"{'Sin reranking':<{col_url}} | "
        f"{'Con reranking':<{col_url}} | "
        f"{'¿Cambió?':<8} | "
        f"{'t_sin':>6} | {'t_con':>6}"
    )
    print(header)
    print("-" * len(header))

    for row in summary:
        q = row["query"][:col_q]
        sin = truncate(row["sin_reranking"], col_url)
        con = truncate(row["con_reranking"], col_url)
        print(
            f"{q:<{col_q}} | "
            f"{sin:<{col_url}} | "
            f"{con:<{col_url}} | "
            f"{row['cambio']:<8} | "
            f"{row['t_sin']:>5.2f}s | "
            f"{row['t_con']:>5.2f}s"
        )

    changed_count = sum(1 for r in summary if r["cambio"] == "SI")
    print(f"\nReranking cambió el top-1 en {changed_count}/{len(summary)} queries.")


if __name__ == "__main__":
    main()
