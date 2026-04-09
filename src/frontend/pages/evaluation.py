"""
Página Streamlit: resultados de evaluación formal del sistema RAG.

Lee data/eval_results.json (generado por notebooks/rag_evaluation.ipynb)
y visualiza métricas de faithfulness y factuality con y sin reranking.
Acceso protegido con la misma contraseña que monitoring.py.

Streamlit detecta automáticamente este archivo en pages/ y lo agrega
como navegación en el sidebar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    st.title("🧪 Evaluación RAG — Bancolombia")
    st.markdown("Acceso restringido al equipo interno.")
    pwd = st.text_input("🔐 Contraseña de acceso", type="password")
    if pwd == MONITORING_PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Contraseña incorrecta. Intenta de nuevo.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Bancolombia RAG")
    st.caption("Evaluación formal del pipeline")
    st.divider()

    st.subheader("⚙️ Fuente de datos")
    eval_path_display = os.getenv("EVAL_RESULTS_PATH", "data/eval_results.json")
    st.code(eval_path_display, language=None)

    st.divider()
    st.subheader("📖 Cómo ejecutar")
    st.markdown(
        "1. Asegúrate de tener ChromaDB con datos\n"
        "2. Ejecuta el notebook:\n"
        "   `notebooks/rag_evaluation.ipynb`\n"
        "3. Regresa a esta página y recarga"
    )

    st.divider()
    st.page_link("app.py", label="← Ir al Chat")

# ──────────────────────────────────────────────────────────────────────────────
# Sección 1 — Header y carga de datos
# ──────────────────────────────────────────────────────────────────────────────

st.title("🧪 Evaluación RAG — Calidad del Sistema")
st.caption("Faithfulness · Factuality Precision · Factuality Recall · F1")

eval_results_path = Path(os.getenv("EVAL_RESULTS_PATH", "data/eval_results.json"))

if not eval_results_path.exists():
    st.warning(
        "⚠️ No hay resultados de evaluación disponibles.\n\n"
        "Ejecuta el notebook `notebooks/rag_evaluation.ipynb` para generar las métricas.\n\n"
        "Asegúrate de tener ChromaDB con datos cargados antes de ejecutarlo."
    )
    st.stop()

try:
    eval_data = json.loads(eval_results_path.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001
    st.error(f"Error leyendo {eval_results_path}: {exc}")
    st.stop()

summary_no_rr = eval_data.get("without_reranking", {}).get("summary", {})
summary_rr = eval_data.get("with_reranking", {}).get("summary", {})
results_no_rr = eval_data.get("without_reranking", {}).get("results", [])
results_rr = eval_data.get("with_reranking", {}).get("results", [])
generated_at = eval_data.get("generated_at", "")

st.caption(f"Evaluación generada: {generated_at[:19].replace('T', ' ') if generated_at else 'N/A'}")
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 2 — KPIs comparativos
# ──────────────────────────────────────────────────────────────────────────────

st.header("📊 KPIs Comparativos: Sin vs Con Reranking")

col_no_rr, col_rr = st.columns(2)

_metrics_map = [
    ("avg_faithfulness", "🎯 Faithfulness"),
    ("avg_factuality_f1", "📐 Factuality F1"),
    ("avg_factuality_precision", "🔍 Precision"),
    ("avg_factuality_recall", "📋 Recall"),
]

with col_no_rr:
    st.subheader("Sin Reranking")
    for key, label in _metrics_map:
        val = summary_no_rr.get(key, 0.0) or 0.0
        st.metric(label, f"{val:.3f}")

with col_rr:
    st.subheader("Con Reranking")
    for key, label in _metrics_map:
        val_rr = summary_rr.get(key, 0.0) or 0.0
        val_no_rr = summary_no_rr.get(key, 0.0) or 0.0
        delta = val_rr - val_no_rr
        sign = "+" if delta >= 0 else ""
        st.metric(label, f"{val_rr:.3f}", delta=f"{sign}{delta:.3f}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 3 — Gráficas
# ──────────────────────────────────────────────────────────────────────────────

st.header("📈 Visualizaciones")

tab_scatter, tab_bar = st.tabs(["F1 vs Faithfulness", "Impacto del Reranking"])

_scatter_path = Path("data/eval_f1_vs_faithfulness.png")
_bar_path = Path("data/eval_reranking_impact.png")

with tab_scatter:
    if _scatter_path.exists():
        st.image(str(_scatter_path), use_column_width=True)
        st.markdown(
            "**Interpretación de cuadrantes:**\n"
            "- ✅ **Ideal** (faith alto, F1 alto): el sistema recupera y responde correctamente\n"
            "- ⚠️ **Alucinación** (faith bajo, F1 alto): responde bien pero sin basarse en los chunks\n"
            "- 🔍 **Retrieval pobre** (faith alto, F1 bajo): fiel al contexto pero el contexto no es el correcto\n"
            "- ❌ **Fallo total** (faith bajo, F1 bajo): ni recupera bien ni responde bien"
        )
    else:
        st.info("Gráfica no disponible. Ejecuta el notebook para generarla.")

with tab_bar:
    if _bar_path.exists():
        st.image(str(_bar_path), use_column_width=True)
        st.markdown(
            "**Cómo leer esta gráfica:**\n"
            "- Barras rojas: resultados sin reranking\n"
            "- Barras verdes: resultados con reranking\n"
            "- Un delta positivo indica que el reranking mejora esa métrica"
        )
    else:
        st.info("Gráfica no disponible. Ejecuta el notebook para generarla.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 4 — Tabla detallada por pregunta
# ──────────────────────────────────────────────────────────────────────────────

st.header("📋 Resultados por Pregunta")

if results_no_rr and results_rr:
    try:
        table_rows = []
        for r_no, r_yes in zip(results_no_rr, results_rr):
            f1_no = r_no.get("factuality_f1") or 0.0
            f1_yes = r_yes.get("factuality_f1") or 0.0
            delta_f1 = round(f1_yes - f1_no, 3)
            table_rows.append(
                {
                    "Pregunta": r_no.get("question", "")[:60] + ("..." if len(r_no.get("question", "")) > 60 else ""),
                    "F1 sin RR": round(f1_no, 3),
                    "F1 con RR": round(f1_yes, 3),
                    "Faith sin RR": round(r_no.get("faithfulness", 0.0), 3),
                    "Faith con RR": round(r_yes.get("faithfulness", 0.0), 3),
                    "Delta F1": delta_f1,
                }
            )

        detail_df = pd.DataFrame(table_rows)

        def _color_delta(val: float):
            if val > 0.05:
                return "background-color: #d4edda"
            if val < -0.05:
                return "background-color: #f8d7da"
            return "background-color: #fff3cd"

        styled = detail_df.style.applymap(_color_delta, subset=["Delta F1"])
        st.dataframe(styled, hide_index=True, use_container_width=True)

        csv_bytes = detail_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Exportar tabla CSV",
            data=csv_bytes,
            file_name="eval_results_detail.csv",
            mime="text/csv",
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Error construyendo tabla: {exc}")
else:
    st.info("No hay resultados disponibles aún.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Sección 5 — Diagnóstico del sistema
# ──────────────────────────────────────────────────────────────────────────────

st.header("🔬 Diagnóstico del Sistema")

try:
    high_faith_low_fact = summary_rr.get("high_faith_low_fact", [])
    low_faith_high_fact = summary_rr.get("low_faith_high_fact", [])

    if high_faith_low_fact:
        st.warning("⚠️ **Posible problema de retrieval**")
        st.markdown(
            "Estas preguntas tienen alta faithfulness pero baja factualidad — "
            "los chunks recuperados no contienen la información correcta sobre el tema:"
        )
        for q in high_faith_low_fact:
            st.markdown(f"- {q}")

    if low_faith_high_fact:
        st.error("🚨 **Posible alucinación detectada**")
        st.markdown(
            "Estas preguntas tienen alta factualidad pero baja faithfulness — "
            "el agente genera información correcta pero que no está en los chunks recuperados:"
        )
        for q in low_faith_high_fact:
            st.markdown(f"- {q}")

    if not high_faith_low_fact and not low_faith_high_fact:
        st.success("✅ Sistema RAG saludable — sin anomalías detectadas en el conjunto de evaluación.")

    # Estadísticas adicionales
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        total = summary_rr.get("total_evaluated", 0)
        gt_avail = summary_rr.get("gt_available", 0)
        st.metric("Preguntas evaluadas", total)
        st.metric("Con ground truth", gt_avail)
    with col_b:
        best = summary_rr.get("best_question", "")
        st.markdown("**Mejor pregunta (mayor F1):**")
        st.caption(best[:80] + ("..." if len(best) > 80 else "") if best else "N/A")
    with col_c:
        worst = summary_rr.get("worst_question", "")
        st.markdown("**Pregunta más difícil (menor F1):**")
        st.caption(worst[:80] + ("..." if len(worst) > 80 else "") if worst else "N/A")

except Exception as exc:  # noqa: BLE001
    st.warning(f"Error cargando diagnóstico: {exc}")
