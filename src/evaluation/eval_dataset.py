"""
Dataset de evaluación formal del sistema RAG — 15 preguntas curadas.

Cubre todas las categorías de la KB: cuentas, tarjetas de crédito,
tarjetas débito, giros. Incluye 2 preguntas fuera de scope para
verificar que el sistema no alucina ante temas no cubiertos.

Uso:
    from src.evaluation.eval_dataset import EVAL_DATASET
    results = evaluator.evaluate_dataset(EVAL_DATASET)
"""

from __future__ import annotations

EVAL_DATASET: list[dict] = [
    # ── CUENTAS (4 preguntas) ────────────────────────────────────────────────
    {
        "question": "¿Qué es la Cuenta AFC y cuáles son sus beneficios tributarios?",
        "ground_truth_url": "https://www.bancolombia.com/personas/cuentas/vivienda/cuenta-afc",
        "expected_category": "cuentas",
        "key_facts": ["ahorro vivienda", "beneficio tributario", "deducción renta"],
    },
    {
        "question": "¿Cuáles son las características de la Cuenta Corriente Bancolombia?",
        "ground_truth_url": "https://www.bancolombia.com/personas/cuentas/ahorros-y-corriente/cuenta-corriente",
        "expected_category": "cuentas",
        "key_facts": ["chequera", "cupo sobregiro", "empresas"],
    },
    {
        "question": "¿Qué es la cuenta Banconautas y para quién está diseñada?",
        "ground_truth_url": "https://www.bancolombia.com/personas/cuentas/ahorros-y-corriente/banconautas",
        "expected_category": "cuentas",
        "key_facts": ["niños", "menores de edad", "educación financiera"],
    },
    {
        "question": "¿Qué beneficios tiene la Cuenta Nómina de Bancolombia?",
        "ground_truth_url": "https://www.bancolombia.com/personas/cuentas/ahorros-y-corriente/cuenta-nomina",
        "expected_category": "cuentas",
        "key_facts": ["salario", "nómina", "empleados"],
    },
    # ── TARJETAS DE CRÉDITO (4 preguntas) ────────────────────────────────────
    {
        "question": "¿Cuáles son los beneficios exclusivos de la tarjeta Infinite Visa?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-de-credito/visa/infinite",
        "expected_category": "tarjetas-de-credito",
        "key_facts": ["Priority Pass", "seguro viaje", "ingresos altos"],
    },
    {
        "question": "¿Qué ofrece la tarjeta Mastercard Platinum de Bancolombia?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-de-credito/mastercard/platinum",
        "expected_category": "tarjetas-de-credito",
        "key_facts": ["beneficios viaje", "seguro médico exterior", "asistencia"],
    },
    {
        "question": "¿Qué es la tarjeta Mastercard Joven y cuáles son sus requisitos?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-de-credito/mastercard/joven",
        "expected_category": "tarjetas-de-credito",
        "key_facts": ["jóvenes", "universitarios", "primer crédito"],
    },
    {
        "question": "¿Qué beneficios tiene la tarjeta LifeMiles Visa de Bancolombia?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-de-credito/visa/lifemiles",
        "expected_category": "tarjetas-de-credito",
        "key_facts": ["millas", "Avianca", "viajes", "acumulación"],
    },
    # ── TARJETAS DÉBITO (3 preguntas) ─────────────────────────────────────────
    {
        "question": "¿Cuáles son los beneficios de la tarjeta débito Black Bancolombia?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-debito/black",
        "expected_category": "tarjetas-debito",
        "key_facts": ["seguro viaje", "pago contactless", "protección fraude"],
    },
    {
        "question": "¿Qué es la tarjeta débito con funcionalidad de transporte?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-debito/transporte",
        "expected_category": "tarjetas-debito",
        "key_facts": ["transporte público", "recarga", "SITP", "metro"],
    },
    {
        "question": "¿Qué beneficios tiene la tarjeta débito Preferencial?",
        "ground_truth_url": "https://www.bancolombia.com/personas/tarjetas-debito/preferencial",
        "expected_category": "tarjetas-debito",
        "key_facts": ["clientes preferencial", "beneficios exclusivos"],
    },
    # ── GIROS (2 preguntas) ──────────────────────────────────────────────────
    {
        "question": "¿Cómo funciona el servicio de giros internacionales con Zaswin?",
        "ground_truth_url": "https://www.bancolombia.com/personas/giros/internacionales/remesas/zaswin",
        "expected_category": "giros",
        "key_facts": ["Estados Unidos", "remesas", "envío dinero", "cuenta Bancolombia"],
    },
    {
        "question": "¿Cómo puede un colombiano en el exterior abrir una cuenta en Bancolombia?",
        "ground_truth_url": (
            "https://www.bancolombia.com/personas/cuentas/desde-el-exterior/cuenta-colombianos-en-el-exterior"
        ),
        "expected_category": "cuentas",
        "key_facts": ["exterior", "colombianos", "cuenta dólares", "requisitos"],
    },
    # ── OUT OF SCOPE (2 preguntas — miden que el sistema NO alucina) ─────────
    {
        "question": "¿Cuál es la tasa de cambio del dólar hoy?",
        "ground_truth_url": None,
        "expected_category": None,
        "key_facts": [],
    },
    {
        "question": "¿Cómo puedo invertir en acciones de la Bolsa de Valores?",
        "ground_truth_url": None,
        "expected_category": None,
        "key_facts": [],
    },
]
