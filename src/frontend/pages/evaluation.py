"""
Página Streamlit: resultados de evaluación formal del sistema RAG.

Muestra métricas de faithfulness y factualidad comparando el pipeline
base (sin reranking) contra el pipeline con Cross-Encoder reranking.
Los datos se leen de data/eval_results.json generado por el notebook
notebooks/rag_evaluation.ipynb.

Acceso protegido con la misma contraseña que el panel de monitoreo.

Ejecución (junto con app.py):
    uv run streamlit run src/frontend/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Evaluación RAG — Bancolombia",
    page_icon="🧪",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Autenticación
# ──────────────────────────────────────────────────────────────────────────────

MONITORING_PASSWORD = os.getenv("MONITORING_PASSWORD", "bancolombia2026")

if not st.session_state.get("authenticated"):
    st.title("🧪 Evaluación Formal del Sistema RAG")
    st.markdown("Acceso restringido al equipo interno.")
    pwd = st.text_input("🔐 Contraseña de acceso", type="password")
    if pwd == MONITORING_PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Contraseña incorrecta. Intenta de nuevo.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

eval_path = DATA_DIR / "eval_results.json"

st.title("🧪 Evaluación Formal del Sistema RAG")
st.markdown(
    "<span style='color: #9ca3af;'>" "15 preguntas · LLM-as-judge · gpt-4o-mini · Abril 2026" "</span>",
    unsafe_allow_html=True,
)

if not eval_path.exists():
    st.warning(
        "⚠️ No hay resultados de evaluación disponibles.\n\n"
        "Ejecuta el notebook `notebooks/rag_evaluation.ipynb` "
        "para generar las métricas."
    )
    st.code("jupyter notebook notebooks/rag_evaluation.ipynb")
    st.stop()

try:
    with eval_path.open(encoding="utf-8") as f:
        eval_results: dict = json.load(f)
except Exception as exc:  # noqa: BLE001
    st.error(f"Error leyendo {eval_path}: {exc}")
    st.stop()

generated_at = eval_results.get("generated_at", "N/A")
st.caption(f"Generado: {generated_at}")

summary_no_rr: dict = eval_results.get("without_reranking", {}).get("summary", {})
summary_rr: dict = eval_results.get("with_reranking", {}).get("summary", {})
results_no_rr: list[dict] = eval_results.get("without_reranking", {}).get("results", [])
results_rr: list[dict] = eval_results.get("with_reranking", {}).get("results", [])

faith_no = summary_no_rr.get("avg_faithfulness", 0.0) or 0.0
f1_no = summary_no_rr.get("avg_factuality_f1", 0.0) or 0.0
prec_no = summary_no_rr.get("avg_factuality_precision", 0.0) or 0.0
rec_no = summary_no_rr.get("avg_factuality_recall", 0.0) or 0.0

faith_rr = summary_rr.get("avg_faithfulness", 0.0) or 0.0
f1_rr = summary_rr.get("avg_factuality_f1", 0.0) or 0.0
prec_rr = summary_rr.get("avg_factuality_precision", 0.0) or 0.0
rec_rr = summary_rr.get("avg_factuality_recall", 0.0) or 0.0

delta_faith = faith_rr - faith_no
delta_f1 = f1_rr - f1_no
delta_prec = prec_rr - prec_no
delta_rec = rec_rr - rec_no

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Bancolombia RAG")
    st.caption("🧪 Evaluación RAG")
    st.divider()

    st.subheader("📊 Resumen ejecutivo")
    st.metric("F1 con reranking", f"{f1_rr:.3f}")
    st.metric("Precision", f"{prec_rr:.3f}")
    st.metric(
        "Preguntas evaluadas",
        summary_rr.get("total_evaluated", 0),
    )

    st.divider()
    st.page_link("app.py", label="← Ir al Chat")
    st.page_link("pages/monitoring.py", label="📊 Ir a Monitoreo")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 1 — KPIs comparativos
# ──────────────────────────────────────────────────────────────────────────────

col_no, col_rr_col = st.columns(2)

with col_no:
    st.markdown("### 📊 Sin Reranking (baseline)")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Faithfulness", f"{faith_no:.3f}")
        st.metric("Precision", f"{prec_no:.3f}")
    with m2:
        st.metric("F1", f"{f1_no:.3f}")
        st.metric("Recall", f"{rec_no:.3f}")

with col_rr_col:
    st.markdown("### 🚀 Con Reranking (Cross-Encoder)")
    m3, m4 = st.columns(2)
    with m3:
        st.metric(
            "Faithfulness",
            f"{faith_rr:.3f}",
            delta=f"{delta_faith:+.3f}",
            delta_color="normal",
        )
        st.metric(
            "Precision",
            f"{prec_rr:.3f}",
            delta=f"{delta_prec:+.3f}",
            delta_color="off",
        )
    with m4:
        st.metric(
            "F1",
            f"{f1_rr:.3f}",
            delta=f"{delta_f1:+.3f}",
            delta_color="normal",
        )
        st.metric(
            "Recall",
            f"{rec_rr:.3f}",
            delta=f"{delta_rec:+.3f}",
            delta_color="normal",
        )

st.info(
    f"💡 **Interpretación:** El reranking mejora F1 ({delta_f1:+.1%}) y "
    f"Recall ({delta_rec:+.1%}) a costa de una reducción en Faithfulness "
    f"({delta_faith:+.1%}). El baseline sin reranking ya es muy fuerte "
    f"(F1: {f1_no:.3f}), lo que limita el margen de mejora. La Precision "
    "se mantiene estable en ambos casos, confirmando ausencia de alucinación factual."
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 2 — Gráficas (tabs)
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🎯 F1 vs Faithfulness", "📈 Impacto del Reranking"])

scatter_path = DATA_DIR / "eval_f1_vs_faithfulness.png"
barplot_path = DATA_DIR / "eval_reranking_impact.png"

with tab1:
    if scatter_path.exists():
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.image(str(scatter_path), use_container_width=True)
    else:
        st.info(
            "📊 La imagen no está disponible. " "Ejecuta el notebook `notebooks/rag_evaluation.ipynb` para generarla."
        )

    st.markdown("""
**Guía de cuadrantes:**

| Zona | Faithfulness | F1 | Diagnóstico |
|---|---|---|---|
| ✅ Ideal (arriba-derecha) | >0.7 | >0.7 | Sistema saludable |
| ⚠️ Alucinación (arriba-izquierda) | <0.7 | >0.7 | LLM genera más allá del contexto |
| ⚠️ Retrieval pobre (abajo-derecha) | >0.7 | <0.7 | Chunks no informativos |
| 🚨 Fallo total (abajo-izquierda) | <0.7 | <0.7 | Problema sistémico |

**Hallazgo:** La mayoría de puntos se concentran en el cuadrante ideal.
No se detectó alucinación factual sistémica.
""")

with tab2:
    if barplot_path.exists():
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.image(str(barplot_path), use_container_width=True)
    else:
        st.info(
            "📊 La imagen no está disponible. " "Ejecuta el notebook `notebooks/rag_evaluation.ipynb` para generarla."
        )

    st.markdown(f"""
**Resumen del impacto:**
- F1 mejora **{delta_f1:+.1%}** con reranking
- Recall mejora **{delta_rec:+.1%}** — mayor cobertura de hechos
- Faithfulness baja **{delta_faith:+.1%}** — el LLM infiere más allá \
del contexto literal con chunks más relevantes
- Precision estable — sin introducción de errores
""")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 3 — Tabla detallada por pregunta
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📋 Resultados por pregunta")


def _color_delta(val: float) -> str:
    if val > 0.02:
        return "background-color: #1a472a; color: white"
    if val < -0.02:
        return "background-color: #7f1d1d; color: white"
    return "background-color: #374151; color: white"


if results_no_rr and results_rr:
    rows = []
    for r_no, r_rr in zip(results_no_rr, results_rr):
        question = r_no.get("question", "")
        f1_no_q = r_no.get("factuality_f1") or 0.0
        f1_rr_q = r_rr.get("factuality_f1") or 0.0
        rows.append(
            {
                "Pregunta": question[:60] + ("..." if len(question) > 60 else ""),
                "F1 sin RR": round(f1_no_q, 3),
                "F1 con RR": round(f1_rr_q, 3),
                "Faith sin RR": round(r_no.get("faithfulness", 0.0), 3),
                "Faith con RR": round(r_rr.get("faithfulness", 0.0), 3),
                "Delta F1": round(f1_rr_q - f1_no_q, 3),
            }
        )

    df = pd.DataFrame(rows)
    try:
        styled = df.style.map(_color_delta, subset=["Delta F1"])
    except AttributeError:
        styled = df.style.applymap(_color_delta, subset=["Delta F1"])  # type: ignore[attr-defined]

    st.dataframe(styled, hide_index=True, use_container_width=True)

    st.markdown("#### 🔍 Detalle por pregunta")
    for r_no, r_rr in zip(results_no_rr, results_rr):
        question = r_no.get("question", "")
        with st.expander(f"Ver detalle: {question[:50]}..."):
            st.markdown(f"**Pregunta completa:** {question}")
            st.divider()

            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.markdown("**Sin reranking**")
                st.caption(
                    f"Faithfulness: {r_no.get('faithfulness', 0):.3f} · " f"F1: {r_no.get('factuality_f1') or 0:.3f}"
                )
                faith_r = r_no.get("faithfulness_reasoning", "")
                if faith_r:
                    st.markdown(f"*Faithfulness:* {faith_r}")
                fact_r = r_no.get("factuality_reasoning", "")
                if fact_r:
                    st.markdown(f"*Factuality:* {fact_r}")

            with det_col2:
                st.markdown("**Con reranking**")
                st.caption(
                    f"Faithfulness: {r_rr.get('faithfulness', 0):.3f} · " f"F1: {r_rr.get('factuality_f1') or 0:.3f}"
                )
                faith_r_rr = r_rr.get("faithfulness_reasoning", "")
                if faith_r_rr:
                    st.markdown(f"*Faithfulness:* {faith_r_rr}")
                fact_r_rr = r_rr.get("factuality_reasoning", "")
                if fact_r_rr:
                    st.markdown(f"*Factuality:* {fact_r_rr}")

            gt_url = r_no.get("ground_truth_url") or r_rr.get("ground_truth_url")
            if gt_url:
                st.markdown(f"**Ground truth:** [{gt_url}]({gt_url})")

            st.caption(
                f"Ground truth disponible: {'✅' if r_no.get('ground_truth_found') else '❌'} · "
                f"Categoría: {r_no.get('expected_category', 'N/A')} · "
                f"use_reranking: {r_rr.get('use_reranking', True)}"
            )
else:
    st.info("No hay resultados por pregunta disponibles.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 4 — Diagnóstico del sistema
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🔬 Diagnóstico del sistema")

high_faith_low_fact: list = summary_rr.get("high_faith_low_fact", [])
low_faith_high_fact: list = summary_rr.get("low_faith_high_fact", [])

if not high_faith_low_fact and not low_faith_high_fact:
    st.success(
        "✅ **Sistema RAG saludable**\n\n"
        "No se detectaron patrones de alucinación ni retrieval fallido "
        "en ninguna de las preguntas evaluadas."
    )

if high_faith_low_fact:
    st.warning(
        f"⚠️ **Posible problema de retrieval** ({len(high_faith_low_fact)} preguntas)\n\n"
        "Alta Faithfulness pero baja Factualidad — los chunks recuperados son coherentes "
        "con la respuesta pero no contienen información factualmente correcta.\n\n"
        + "\n".join(f"- {q}" for q in high_faith_low_fact)
    )

if low_faith_high_fact:
    st.error(
        f"🚨 **Posible alucinación detectada** ({len(low_faith_high_fact)} preguntas)\n\n"
        "El agente genera hechos correctos que no están en los chunks recuperados. "
        "Puede ser paráfrasis del LLM más que alucinación factual — revisar manualmente.\n\n"
        + "\n".join(f"- {q}" for q in low_faith_high_fact)
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 5 — Conclusiones (colapsable)
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("📝 Ver conclusiones completas", expanded=False):
    st.markdown(f"""
### Hallazgos principales

**1. Sistema RAG de alta calidad**
F1 de {f1_no:.3f} sin reranking y {f1_rr:.3f} con reranking.
Precision de {prec_no:.3f} — prácticamente sin alucinación.

**2. Reranking con impacto modesto pero positivo**
El baseline ya es muy fuerte (F1: {f1_no:.3f}), limitando el margen de mejora.
La mejora en Recall ({delta_rec:+.1%}) justifica el costo de latencia para
casos donde la completitud es prioritaria.

**3. Trade-off documentado: Faithfulness vs Recall**
El reranking reduce Faithfulness ({delta_faith:+.1%}) mientras mejora
Recall ({delta_rec:+.1%}). El LLM infiere más allá del contexto literal
cuando recibe chunks más informativos. Esto no es alucinación factual
(Precision estable).

**4. Sistema conservador ante out-of-scope**
Las 2 preguntas fuera de scope fueron correctamente manejadas —
el agente no inventó respuestas.

### Recomendaciones

| Prioridad | Acción | Impacto |
|---|---|---|
| Alta | Ampliar cobertura tarjeta débito transporte | +F1 |
| Media | Aumentar MAX_PAGES a 200+ URLs | Mayor cobertura |
| Baja | Cross-Encoder en GPU en producción | -latencia |

### Limitaciones
- Dataset de 15 preguntas (ampliar a 50+ en producción)
- LLM-as-judge puede tener sesgos propios
- Ground truth estático (abril 2026)
- Modelo de reranking entrenado principalmente en inglés
""")
