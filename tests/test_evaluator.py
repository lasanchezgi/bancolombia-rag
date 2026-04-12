"""
Tests unitarios para RAGEvaluator.

Usa mocks de OpenAI para evitar llamadas reales a la API.
Todos los tests son deterministas y no requieren servicios externos.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.eval_dataset import EVAL_DATASET
from src.evaluation.evaluator import RAGEvaluator

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def evaluator(tmp_path: Path) -> RAGEvaluator:
    """RAGEvaluator con cliente OpenAI mockeado y data_raw_path en directorio temporal."""
    ev = RAGEvaluator.__new__(RAGEvaluator)
    ev.client = MagicMock()
    ev.data_raw_path = str(tmp_path)
    return ev


def _make_judge_response(payload: dict) -> MagicMock:
    """Construye un mock de respuesta de OpenAI que retorna JSON válido."""
    response_mock = MagicMock()
    response_mock.choices[0].message.content = json.dumps(payload)
    return response_mock


# ──────────────────────────────────────────────────────────────────────────────
# Tests de _load_ground_truth
# ──────────────────────────────────────────────────────────────────────────────


def test_load_ground_truth_existing_url(evaluator: RAGEvaluator, tmp_path: Path) -> None:
    """Debe retornar el campo 'text' del JSON dado un URL que existe en data/raw/."""
    url = "https://www.bancolombia.com/personas/cuentas/vivienda/cuenta-afc"
    filename = "personas_cuentas_vivienda_cuenta_afc.json"
    (tmp_path / filename).write_text(
        json.dumps({"url": url, "title": "Cuenta AFC", "text": "Texto de la cuenta AFC"}),
        encoding="utf-8",
    )
    result = evaluator._load_ground_truth(url)
    assert result == "Texto de la cuenta AFC"


def test_load_ground_truth_missing_url(evaluator: RAGEvaluator) -> None:
    """Debe retornar None sin lanzar excepción cuando el archivo no existe."""
    url = "https://www.bancolombia.com/personas/cuentas/inexistente/producto"
    result = evaluator._load_ground_truth(url)
    assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests de _call_judge
# ──────────────────────────────────────────────────────────────────────────────


def test_call_judge_parses_json_response(evaluator: RAGEvaluator) -> None:
    """Debe parsear correctamente un JSON válido retornado por el juez."""
    payload = {"score": 0.85, "supported_claims": 3, "total_claims": 4, "reasoning": "OK"}
    evaluator.client.chat.completions.create.return_value = _make_judge_response(payload)

    result = evaluator._call_judge("some prompt")

    assert result["score"] == 0.85
    assert result["supported_claims"] == 3
    assert result["total_claims"] == 4


def test_call_judge_handles_invalid_json(evaluator: RAGEvaluator) -> None:
    """Debe retornar dict con scores 0.0 cuando el juez retorna texto no-JSON."""
    bad_response = MagicMock()
    bad_response.choices[0].message.content = "Esto no es JSON válido"
    evaluator.client.chat.completions.create.return_value = bad_response

    result = evaluator._call_judge("some prompt")

    assert result["score"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests de evaluate_single
# ──────────────────────────────────────────────────────────────────────────────


def test_evaluate_single_with_ground_truth(evaluator: RAGEvaluator, tmp_path: Path) -> None:
    """F1 debe ser 2*P*R/(P+R) cuando hay ground truth disponible."""
    url = "https://www.bancolombia.com/personas/cuentas/vivienda/cuenta-afc"
    filename = "personas_cuentas_vivienda_cuenta_afc.json"
    (tmp_path / filename).write_text(
        json.dumps({"text": "Texto oficial de la cuenta AFC"}),
        encoding="utf-8",
    )

    faith_payload = {"score": 0.9, "supported_claims": 9, "total_claims": 10, "reasoning": "OK"}
    fact_payload = {
        "precision": 0.8,
        "recall": 0.6,
        "correct_facts": 4,
        "total_facts_response": 5,
        "gt_facts_covered": 3,
        "total_gt_facts": 5,
        "reasoning": "OK",
    }

    evaluator.client.chat.completions.create.side_effect = [
        _make_judge_response(faith_payload),
        _make_judge_response(fact_payload),
    ]

    result = evaluator.evaluate_single(
        question="¿Qué es la Cuenta AFC?",
        response="La Cuenta AFC es para ahorro de vivienda.",
        retrieved_chunks=[{"text": "texto del chunk", "url": "http://example.com"}],
        ground_truth_url=url,
    )

    assert result["faithfulness"] == pytest.approx(0.9)
    assert result["factuality_precision"] == pytest.approx(0.8)
    assert result["factuality_recall"] == pytest.approx(0.6)
    expected_f1 = 2 * 0.8 * 0.6 / (0.8 + 0.6)
    assert result["factuality_f1"] == pytest.approx(expected_f1)
    assert result["ground_truth_found"] is True


def test_evaluate_single_without_ground_truth(evaluator: RAGEvaluator) -> None:
    """Cuando ground_truth_url=None: faithfulness calculado, factuality=None."""
    faith_payload = {"score": 0.75, "supported_claims": 3, "total_claims": 4, "reasoning": "OK"}
    evaluator.client.chat.completions.create.return_value = _make_judge_response(faith_payload)

    result = evaluator.evaluate_single(
        question="¿Cuál es la tasa de cambio hoy?",
        response="No tengo información sobre eso.",
        retrieved_chunks=[],
        ground_truth_url=None,
    )

    assert result["faithfulness"] == pytest.approx(0.75)
    assert result["factuality_precision"] is None
    assert result["factuality_recall"] is None
    assert result["factuality_f1"] is None
    assert result["ground_truth_found"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Tests de compute_summary
# ──────────────────────────────────────────────────────────────────────────────


def _make_result(
    question: str,
    faithfulness: float,
    f1: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
) -> dict:
    """Helper para construir un resultado de evaluación."""
    return {
        "question": question,
        "faithfulness": faithfulness,
        "factuality_f1": f1,
        "factuality_precision": precision,
        "factuality_recall": recall,
        "supported_claims": 0,
        "total_claims": 0,
        "ground_truth_found": f1 is not None,
    }


def test_compute_summary_averages(evaluator: RAGEvaluator) -> None:
    """avg_faithfulness debe ser la media de los valores de faithfulness."""
    results = [
        _make_result("Q1", faithfulness=0.6, f1=0.5, precision=0.5, recall=0.5),
        _make_result("Q2", faithfulness=0.8, f1=0.7, precision=0.7, recall=0.7),
        _make_result("Q3", faithfulness=1.0, f1=0.9, precision=0.9, recall=0.9),
    ]
    summary = evaluator.compute_summary(results)

    assert summary["avg_faithfulness"] == pytest.approx((0.6 + 0.8 + 1.0) / 3)
    assert summary["total_evaluated"] == 3
    assert summary["gt_available"] == 3


def test_compute_summary_f1_calculation(evaluator: RAGEvaluator) -> None:
    """avg_factuality_f1 debe ser la media de los F1 disponibles."""
    results = [
        _make_result("Q1", faithfulness=0.7, f1=0.4, precision=0.4, recall=0.4),
        _make_result("Q2", faithfulness=0.8, f1=0.6, precision=0.6, recall=0.6),
        _make_result("Q3", faithfulness=0.9, f1=None),  # sin GT
    ]
    summary = evaluator.compute_summary(results)

    assert summary["avg_factuality_f1"] == pytest.approx((0.4 + 0.6) / 2)
    assert summary["gt_available"] == 2


def test_high_faith_low_fact_detection(evaluator: RAGEvaluator) -> None:
    """Pregunta con faithfulness > 0.7 y F1 < 0.5 debe aparecer en high_faith_low_fact."""
    results = [
        _make_result("Pregunta con retrieval pobre", faithfulness=0.9, f1=0.3, precision=0.3, recall=0.3),
        _make_result("Pregunta normal", faithfulness=0.8, f1=0.8, precision=0.8, recall=0.8),
    ]
    summary = evaluator.compute_summary(results)

    assert "Pregunta con retrieval pobre" in summary["high_faith_low_fact"]
    assert "Pregunta normal" not in summary["high_faith_low_fact"]


def test_low_faith_high_fact_detection(evaluator: RAGEvaluator) -> None:
    """Pregunta con faithfulness < 0.5 y F1 > 0.7 debe aparecer en low_faith_high_fact."""
    results = [
        _make_result("Pregunta con alucinación", faithfulness=0.2, f1=0.8, precision=0.8, recall=0.8),
        _make_result("Pregunta normal", faithfulness=0.9, f1=0.9, precision=0.9, recall=0.9),
    ]
    summary = evaluator.compute_summary(results)

    assert "Pregunta con alucinación" in summary["low_faith_high_fact"]
    assert "Pregunta normal" not in summary["low_faith_high_fact"]


# ──────────────────────────────────────────────────────────────────────────────
# Tests adicionales
# ──────────────────────────────────────────────────────────────────────────────


def test_out_of_scope_questions_handled(evaluator: RAGEvaluator) -> None:
    """Preguntas con ground_truth_url=None no deben lanzar excepción."""
    faith_payload = {"score": 0.5, "supported_claims": 1, "total_claims": 2, "reasoning": "OK"}
    evaluator.client.chat.completions.create.return_value = _make_judge_response(faith_payload)

    result = evaluator.evaluate_single(
        question="¿Cuál es la tasa de cambio del dólar hoy?",
        response="No tengo información sobre tasas de cambio.",
        retrieved_chunks=[],
        ground_truth_url=None,
    )

    assert result["factuality_f1"] is None
    assert result["factuality_precision"] is None
    assert result["factuality_recall"] is None
    assert result["faithfulness"] == pytest.approx(0.5)


def test_eval_dataset_has_15_questions() -> None:
    """El dataset de evaluación debe tener exactamente 15 preguntas."""
    assert len(EVAL_DATASET) == 15


def test_eval_dataset_last_two_are_out_of_scope() -> None:
    """Las últimas 2 preguntas deben tener ground_truth_url=None (fuera de scope)."""
    assert EVAL_DATASET[-1]["ground_truth_url"] is None
    assert EVAL_DATASET[-2]["ground_truth_url"] is None


def test_compute_summary_empty_results(evaluator: RAGEvaluator) -> None:
    """compute_summary con lista vacía no debe lanzar excepción y retorna ceros."""
    summary = evaluator.compute_summary([])
    assert summary["avg_faithfulness"] == 0.0
    assert summary["total_evaluated"] == 0
