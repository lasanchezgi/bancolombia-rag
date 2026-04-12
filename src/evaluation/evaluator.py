"""
Evaluador formal del sistema RAG usando LLM-as-judge.

Mide faithfulness (¿la respuesta está soportada por los chunks recuperados?)
y factuality (¿los hechos son correctos respecto al ground truth oficial?).
Cada evaluación se puede correr con y sin reranking para comparar su impacto.

Ground truth: archivos JSON en data/raw/
Juez: gpt-4o-mini (~$0.002 por pregunta)
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import openai

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluador formal del sistema RAG usando LLM-as-judge.

    Métricas implementadas:
    - Faithfulness: claims de la respuesta soportados por chunks
    - Factuality Precision: hechos correctos / total hechos respuesta
    - Factuality Recall: hechos GT cubiertos / total hechos GT
    - Factuality F1: 2 * (P * R) / (P + R)

    Cada métrica se evalúa con y sin reranking para comparación.
    Ground truth: archivos JSON en data/raw/
    """

    FAITHFULNESS_PROMPT = """
Eres un evaluador experto de sistemas RAG.
Evalúa si la respuesta del agente está completamente soportada
por el contexto recuperado.

CONTEXTO RECUPERADO (chunks del sistema RAG):
{context}

RESPUESTA DEL AGENTE:
{response}

INSTRUCCIONES:
1. Identifica cada afirmación factual en la respuesta
2. Verifica si cada afirmación está soportada por el contexto
3. Calcula: afirmaciones soportadas / total afirmaciones

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"score": 0.0, "supported_claims": 0, "total_claims": 0,
  "reasoning": "explicación breve en español"}}

Score 1.0 = todas las afirmaciones están en el contexto
Score 0.0 = ninguna afirmación está en el contexto
"""

    FACTUALITY_PROMPT = """
Eres un evaluador experto de sistemas RAG.
Evalúa la calidad factual de la respuesta comparándola con
el texto oficial de la página web de Bancolombia.

RESPUESTA DEL AGENTE:
{response}

GROUND TRUTH (texto oficial de bancolombia.com):
{ground_truth}

INSTRUCCIONES:
Evalúa estas tres dimensiones en escala 0.0 a 1.0:

1. Precision: ¿Qué proporción de los hechos en la respuesta
   son correctos según el ground truth?
   (hechos correctos / total hechos en respuesta)

2. Recall: ¿Qué proporción de los hechos importantes del
   ground truth están cubiertos en la respuesta?
   (hechos GT cubiertos / total hechos importantes en GT)

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"precision": 0.0, "recall": 0.0,
  "correct_facts": 0, "total_facts_response": 0,
  "gt_facts_covered": 0, "total_gt_facts": 0,
  "reasoning": "explicación breve en español"}}
"""

    RESPONSE_SYSTEM_PROMPT = (
        "Eres un asistente virtual especializado en productos y servicios de Bancolombia. "
        "Responde de forma clara y precisa basándote ÚNICAMENTE en el contexto proporcionado. "
        "Si el contexto no contiene información suficiente, indícalo claramente."
    )

    def __init__(self, openai_api_key: str, data_raw_path: str = "data/raw") -> None:
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.data_raw_path = data_raw_path
        logger.info("RAGEvaluator inicializado")

    def _load_ground_truth(self, url: str) -> str | None:
        """
        Carga el texto del ground truth desde data/raw/ dado el URL de la página.

        Convierte URL a filename:
        url.replace("https://www.bancolombia.com/", "")
           .replace("/", "_").replace("-", "_") + ".json"

        Retorna el campo "text" del JSON.
        Retorna None si el archivo no existe.
        """
        filename = url.replace("https://www.bancolombia.com/", "").replace("/", "_").replace("-", "_") + ".json"
        path = Path(self.data_raw_path) / filename
        if not path.exists():
            logger.warning("Ground truth no encontrado: %s", path)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("text") or None
        except Exception as exc:  # noqa: BLE001
            logger.error("Error cargando ground truth %s: %s", path, exc)
            return None

    def _call_judge(self, prompt: str) -> dict:
        """
        Llama a gpt-4o-mini como juez.
        Parsea el JSON de la respuesta.
        En caso de error: retorna dict con scores en 0.0
        Retry 2 veces ante fallos.
        """
        fallback: dict = {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "correct_facts": 0,
            "total_facts_response": 0,
            "gt_facts_covered": 0,
            "total_gt_facts": 0,
            "reasoning": "Error al llamar al juez",
        }
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                # Extraer JSON del contenido (puede haber texto antes/después)
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON encontrado en la respuesta del juez")
                return json.loads(content[start:end])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intento %d/%d fallido en _call_judge: %s", attempt + 1, 3, exc)
        logger.error("_call_judge falló en todos los intentos — retornando fallback")
        return fallback

    def evaluate_single(
        self,
        question: str,
        response: str,
        retrieved_chunks: list[dict],
        ground_truth_url: str | None,
    ) -> dict:
        """
        Evalúa una sola interacción RAG.

        Retorna:
        {
            "question": str,
            "faithfulness": float,
            "faithfulness_reasoning": str,
            "factuality_precision": float | None,
            "factuality_recall": float | None,
            "factuality_f1": float | None,
            "factuality_reasoning": str | None,
            "supported_claims": int,
            "total_claims": int,
            "ground_truth_found": bool
        }

        Si ground_truth no existe: faithfulness calculado,
        factuality scores = None con nota "GT no disponible"
        """
        # Faithfulness — siempre se calcula
        context = "\n\n---\n\n".join(c.get("text", "") for c in retrieved_chunks if c.get("text"))
        if not context:
            context = "(sin contexto recuperado)"

        faith_result = self._call_judge(self.FAITHFULNESS_PROMPT.format(context=context, response=response))
        faithfulness = float(faith_result.get("score", 0.0))
        faithfulness_reasoning = faith_result.get("reasoning", "")
        supported_claims = int(faith_result.get("supported_claims", 0))
        total_claims = int(faith_result.get("total_claims", 0))

        # Factuality — solo si hay ground truth URL
        factuality_precision: float | None = None
        factuality_recall: float | None = None
        factuality_f1: float | None = None
        factuality_reasoning: str | None = None
        ground_truth_found = False

        if ground_truth_url is not None:
            gt_text = self._load_ground_truth(ground_truth_url)
            if gt_text:
                ground_truth_found = True
                fact_result = self._call_judge(
                    self.FACTUALITY_PROMPT.format(response=response, ground_truth=gt_text[:4000])
                )
                factuality_precision = float(fact_result.get("precision", 0.0))
                factuality_recall = float(fact_result.get("recall", 0.0))
                p, r = factuality_precision, factuality_recall
                factuality_f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                factuality_reasoning = fact_result.get("reasoning", "")
            else:
                factuality_reasoning = "GT no disponible"
        else:
            factuality_reasoning = "GT no disponible (pregunta fuera de scope)"

        return {
            "question": question,
            "faithfulness": faithfulness,
            "faithfulness_reasoning": faithfulness_reasoning,
            "factuality_precision": factuality_precision,
            "factuality_recall": factuality_recall,
            "factuality_f1": factuality_f1,
            "factuality_reasoning": factuality_reasoning,
            "supported_claims": supported_claims,
            "total_claims": total_claims,
            "ground_truth_found": ground_truth_found,
        }

    def _search(self, query: str, use_reranking: bool, top_k: int = 5) -> list[dict]:
        """
        Busca chunks relevantes directamente via Embedder + ChromaDB + Reranker opcional.
        Replica la lógica de search_knowledge_base sin overhead del servidor MCP.
        """
        from src.mcp_server.tools import _get_embedder, _get_repository, get_reranker  # noqa: PLC0415

        embedder = _get_embedder()
        repo = _get_repository()
        embedding = embedder.embed_texts([query])[0]
        n_candidates = top_k * 3 if use_reranking else top_k
        raw = repo.query(query_embedding=embedding, top_k=min(n_candidates, 30))
        if not raw:
            return []
        if use_reranking:
            return get_reranker().rerank(query, raw, top_k)
        return raw[:top_k]

    def _generate_response(self, question: str, chunks: list[dict]) -> str:
        """Genera respuesta con gpt-4o-mini dado el contexto de chunks recuperados."""
        context = "\n\n".join(f"[{c.get('url', '')}]\n{c.get('text', '')}" for c in chunks)
        if not context.strip():
            context = "(no se encontró información relevante en la base de conocimiento)"
        messages = [
            {"role": "system", "content": self.RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
        ]
        try:
            resp = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("Error generando respuesta: %s", exc)
            return ""

    def evaluate_dataset(
        self,
        dataset: list[dict],
        use_reranking: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Evalúa el dataset completo.

        Cada item del dataset:
        {
            "question": str,
            "ground_truth_url": str | None,
            "expected_category": str | None,
            "key_facts": list[str]
        }

        Para cada pregunta:
        1. Buscar chunks con use_reranking
        2. Generar respuesta con gpt-4o-mini
        3. Evaluar con evaluate_single

        Retorna lista de resultados con métricas completas.
        """
        results = []
        for i, item in enumerate(dataset):
            if verbose:
                logger.info("Evaluando [%d/%d]: %s", i + 1, len(dataset), item["question"][:50])
            try:
                chunks = self._search(item["question"], use_reranking=use_reranking)
                response = self._generate_response(item["question"], chunks)
                result = self.evaluate_single(
                    question=item["question"],
                    response=response,
                    retrieved_chunks=chunks,
                    ground_truth_url=item.get("ground_truth_url"),
                )
                result["expected_category"] = item.get("expected_category")
                result["use_reranking"] = use_reranking
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error evaluando pregunta %d: %s", i + 1, exc)
                results.append(
                    {
                        "question": item["question"],
                        "faithfulness": 0.0,
                        "faithfulness_reasoning": f"Error: {exc}",
                        "factuality_precision": None,
                        "factuality_recall": None,
                        "factuality_f1": None,
                        "factuality_reasoning": f"Error: {exc}",
                        "supported_claims": 0,
                        "total_claims": 0,
                        "ground_truth_found": False,
                        "expected_category": item.get("expected_category"),
                        "use_reranking": use_reranking,
                    }
                )
        return results

    def compute_summary(self, results: list[dict]) -> dict:
        """
        Calcula métricas agregadas del dataset completo.

        Retorna:
        {
            "avg_faithfulness": float,
            "avg_factuality_precision": float,
            "avg_factuality_recall": float,
            "avg_factuality_f1": float,
            "std_faithfulness": float,
            "std_f1": float,
            "best_question": str,
            "worst_question": str,
            "high_faith_low_fact": list[str],
            "low_faith_high_fact": list[str],
            "total_evaluated": int,
            "gt_available": int
        }
        """
        if not results:
            return {
                "avg_faithfulness": 0.0,
                "avg_factuality_precision": 0.0,
                "avg_factuality_recall": 0.0,
                "avg_factuality_f1": 0.0,
                "std_faithfulness": 0.0,
                "std_f1": 0.0,
                "best_question": "",
                "worst_question": "",
                "high_faith_low_fact": [],
                "low_faith_high_fact": [],
                "total_evaluated": 0,
                "gt_available": 0,
            }

        faithfulness_vals = [r["faithfulness"] for r in results]
        f1_vals = [r["factuality_f1"] for r in results if r.get("factuality_f1") is not None]
        prec_vals = [r["factuality_precision"] for r in results if r.get("factuality_precision") is not None]
        recall_vals = [r["factuality_recall"] for r in results if r.get("factuality_recall") is not None]

        avg_faithfulness = sum(faithfulness_vals) / len(faithfulness_vals)
        avg_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        avg_prec = sum(prec_vals) / len(prec_vals) if prec_vals else 0.0
        avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0

        std_faith = statistics.stdev(faithfulness_vals) if len(faithfulness_vals) >= 2 else 0.0
        std_f1 = statistics.stdev(f1_vals) if len(f1_vals) >= 2 else 0.0

        # Best/worst por F1 (solo donde F1 disponible)
        results_with_f1 = [r for r in results if r.get("factuality_f1") is not None]
        best_question = ""
        worst_question = ""
        if results_with_f1:
            best_question = max(results_with_f1, key=lambda r: r["factuality_f1"])["question"]
            worst_question = min(results_with_f1, key=lambda r: r["factuality_f1"])["question"]

        # Diagnóstico de cuadrantes
        high_faith_low_fact = [
            r["question"] for r in results_with_f1 if r["faithfulness"] > 0.7 and (r["factuality_f1"] or 0.0) < 0.5
        ]
        low_faith_high_fact = [
            r["question"] for r in results_with_f1 if r["faithfulness"] < 0.5 and (r["factuality_f1"] or 0.0) > 0.7
        ]

        return {
            "avg_faithfulness": avg_faithfulness,
            "avg_factuality_precision": avg_prec,
            "avg_factuality_recall": avg_recall,
            "avg_factuality_f1": avg_f1,
            "std_faithfulness": std_faith,
            "std_f1": std_f1,
            "best_question": best_question,
            "worst_question": worst_question,
            "high_faith_low_fact": high_faith_low_fact,
            "low_faith_high_fact": low_faith_high_fact,
            "total_evaluated": len(results),
            "gt_available": len(results_with_f1),
        }
