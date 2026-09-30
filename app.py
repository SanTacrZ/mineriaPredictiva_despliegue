"""Predictor de Diabetes (Pima Indians) — modelo XGBoost.
Ejecutar con el entorno del proyecto: ./env/bin/streamlit run app.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
            "DiabetesPedigreeFunction", "Age"]
LABELS_ES = {
    "Pregnancies": "Embarazos",
    "Glucose": "Glucosa (mg/dL)",
    "BloodPressure": "Presión arterial (mm Hg)",
    "BMI": "IMC (kg/m²)",
    "DiabetesPedigreeFunction": "Antecedente familiar (DPF)",
    "Age": "Edad (años)",
}
# Rangos clínicos razonables para los controles + medianas de imputación del informe
RANGES = {
    "Pregnancies": (0, 17, 2, 1),
    "Glucose": (40, 250, 120, 1),
    "BloodPressure": (30, 130, 72, 1),
    "BMI": (15.0, 60.0, 32.3, 0.1),
    "DiabetesPedigreeFunction": (0.05, 2.50, 0.45, 0.01),
    "Age": (18, 85, 32, 1),
}
MEDIANAS = {"Glucose": 117.0, "BloodPressure": 72.0, "BMI": 32.3}
IMPORTANCIA_XGB = {  # del informe (xgboost, 50 árboles)
    "Glucose": 0.296, "BMI": 0.185, "Age": 0.162,
    "DiabetesPedigreeFunction": 0.128, "BloodPressure": 0.118,
    "Pregnancies": 0.112,
}
UMBRAL = 0.5

st.set_page_config(
    page_title="Predictor de Diabetes | XGBoost",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --teal:#0E7C7B; --navy:#0B2545; --coral:#FF6B6B; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.hero { background: linear-gradient(135deg,#0B2545 0%,#0E7C7B 70%,#17B890 100%);
        color:#fff; border-radius:18px; padding:28px 30px; margin-bottom:18px; }
.hero h1 { margin:0; font-size:2rem; }
.hero p { margin:.4rem 0 0; opacity:.92; }
.chips { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
.chip { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35);
        padding:6px 14px; border-radius:999px; font-size:.82rem; }
.card { background:#fff; border:1px solid #E3EAF0; border-radius:16px; padding:20px 22px;
        box-shadow:0 4px 18px rgba(11,37,69,.06); margin-bottom:16px; }
.badge-ok { background:#E6F7ED; color:#147A3E; border:1px solid #9ADBB4;
            padding:8px 18px; border-radius:999px; font-weight:700; }
.badge-risk { background:#FDECEC; color:#B42318; border:1px solid #F5A3A3;
              padding:8px 18px; border-radius:999px; font-weight:700; }
.small { color:#5B6B7B; font-size:.86rem; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    import pickle

    with open(BASE / "modelo_Xgboost.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_metricas():
    with open(BASE / "results.json", encoding="utf-8") as f:
        res = json.load(f)
    tabla = pd.read_csv(BASE / "resultados_medidas.csv", index_col=0)
    return res, tabla


def predecir(df: pd.DataFrame):
    modelo = cargar_modelo()
    proba = modelo.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= UMBRAL).astype(int)
    return proba, pred


def gauge(prob: float):
    color = "#147A3E" if prob < UMBRAL else "#B42318"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(prob * 100),
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probabilidad de diabetes", "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "steps": [{"range": [0, 50], "color": "#E6F7ED"},
                         {"range": [50, 100], "color": "#FDECEC"}],
               "threshold": {"line": {"color": "#0B2545", "width": 3},
                             "value": 50}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ---------------- HERO ----------------
st.markdown(
    """<div class="hero">
    <h1>🩺 Predictor de Riesgo de Diabetes</h1>
    <p>Minería predictiva en Python · Dataset Pima Indians Diabetes (NIDDK) ·
    Modelo final <b>XGBoost</b> (50 árboles · validación cruzada 10 pliegues · Accuracy test 0.769)</p>
    <div class="chips"><span class="chip">🎯 Accuracy 0.769</span>
    <span class="chip">📈 ROC-AUC 0.807</span><span class="chip">🔁 Recall 0.600</span>
    <span class="chip">🧪 6 variables clínicas</span></div></div>""",
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📝 Datos del paciente")
    st.caption("Ajusta las 6 variables clínicas y pulsa predecir.")
    vals = {}
    for feat in FEATURES:
        lo, hi, dflt, step = RANGES[feat]
        vals[feat] = st.slider(LABELS_ES[feat], min_value=lo, max_value=hi,
                               value=dflt, step=step)
    st.divider()
    umbral = st.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05)
    st.caption(f"Umbral actual: **{umbral:.2f}** (por defecto 0.50)")
    btn = st.button("🔍 Predecir riesgo", type="primary", use_container_width=True)
    st.divider()
    st.caption("Pipeline: `MinMaxScaler` + `XGBClassifier(50 árboles)` · "
               "Imputación por mediana: Glucosa 117 · Presión 72 · IMC 32.3")

tab_pred, tab_mod, tab_evid, tab_datos = st.tabs(
    ["🔮 Predicción", "📊 Modelos comparados", "🖼️ Evidencia", "📁 Datos y metodología"])

# ---------------- TAB PREDICCIÓN ----------------
with tab_pred:
    col1, col2 = st.columns([1.05, 1])
    entrada = pd.DataFrame([vals], columns=FEATURES)
    proba, pred = predecir(entrada)
    p = float(proba[0])
    positivo = p >= umbral

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resultado")
        st.plotly_chart(gauge(p), use_container_width=True)
        if positivo:
            st.markdown('<span class="badge-risk">⚠️ RIESGO ALTO — DIABETES</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-ok">✅ RIESGO BAJO — NO DIABETES</span>',
                        unsafe_allow_html=True)
        st.write(f"**Probabilidad:** `{p:.3f}` · **Umbral:** `{umbral:.2f}` · "
                 f"**Clase:** `{'1 (diabetes)' if positivo else '0 (no diabetes)'}`")
        if p >= 0.75:
            st.warning("Probabilidad alta: se recomienda valoración médica prioritaria.")
        elif p >= 0.5:
            st.info("Zona límite: repetir glucosa en ayunas y controlar IMC/presión.")
        else:
            st.success("Fuera de la zona de riesgo según el modelo. Mantener hábitos saludables.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📦 Predicción por lote (CSV)")
        st.caption("Sube un CSV con columnas: " + ", ".join(FEATURES))
        up = st.file_uploader("Archivo CSV", type="csv", label_visibility="collapsed")
        if up is not None:
            try:
                lote = pd.read_csv(up)
                faltan = [c for c in FEATURES if c not in lote.columns]
                if faltan:
                    st.error(f"Faltan columnas: {faltan}")
                else:
                    pb, pr = predecir(lote)
                    out = lote[FEATURES].copy()
                    out["P(diabetes)"] = np.round(pb, 4)
                    out["Predicción"] = np.where(pr == 1, "DIABETES", "NO DIABETES")
                    st.dataframe(out, use_container_width=True)
                    st.download_button("⬇️ Descargar predicciones",
                                       out.to_csv(index=False).encode(),
                                       "predicciones_lote.csv", "text/csv")
            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Variables ingresadas")
        det = pd.DataFrame({"Variable": [LABELS_ES[f] for f in FEATURES],
                            "Valor": [vals[f] for f in FEATURES],
                            "Importancia XGB": [IMPORTANCIA_XGB[f] for f in FEATURES]})
        st.dataframe(det, use_container_width=True, hide_index=True)
        st.bar_chart(det.set_index("Variable")["Importancia XGB"])
        st.caption("Importancia del modelo ganador: la **Glucosa (0.296)** es la señal dominante, "
                   "seguida del IMC y la Edad.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 Lectura clínica rápida")
        msgs = []
        if vals["Glucose"] >= 140:
            msgs.append("🔴 Glucosa elevada (≥140): principal factor de riesgo del modelo.")
        if vals["BMI"] >= 30:
            msgs.append("🟠 IMC ≥30 (obesidad): segundo factor en importancia.")
        if vals["Age"] >= 45:
            msgs.append("🟡 Edad ≥45: aumenta la probabilidad basal.")
        if not msgs:
            msgs.append("🟢 Valores dentro de rangos moderados.")
        for m in msgs:
            st.write(m)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB MODELOS ----------------
with tab_mod:
    res, tabla = cargar_metricas()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏆 Comparativa (test 30% · GridSearchCV 10 pliegues)")
    st.dataframe(tabla.style.format("{:.3f}").background_gradient(cmap="YlGn"),
                 use_container_width=True)
    mejor = tabla["Accuracy"].idxmax()
    st.success(f"**Mejor modelo: {mejor}** — Accuracy {tabla.loc[mejor, 'Accuracy']:.3f} · "
               f"F1 {tabla.loc[mejor, 'F1']:.3f} · Recall {tabla.loc[mejor, 'Recall']:.3f} · "
               f"ROC-AUC {tabla.loc[mejor, 'ROC-AUC']:.3f}. "
               "La SVM compite en AUC (0.829) pero detecta peor los casos positivos "
               "(recall 0.475 vs 0.600).")
    st.bar_chart(tabla["Accuracy"])
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Matrices de confusión")
        st.image(str(BASE / "figuras" / "m6_confusion.png"), use_container_width=True)
        st.caption("XGBoost: [[128, 21], [32, 48]] — mejor equilibrio en verdaderos positivos.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Curvas ROC")
        st.image(str(BASE / "figuras" / "m7_roc.png"), use_container_width=True)
        st.caption("SVM logra el AUC mayor; XGBoost el mejor compromiso accuracy/recall.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB EVIDENCIA ----------------
with tab_evid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔎 Selección de factores y evidencias del notebook")
    st.write("**Correlación con Outcome:** Glucosa 0.491 · IMC 0.313 · Edad 0.235 · "
             "Embarazos 0.219 · DPF 0.172 · Presión 0.166. Ningún par supera |r|≥0.8 "
             "(máx. Embarazos–Edad ≈0.54): no hay redundancia grave. "
             "Se eliminaron `Insulin` (48.7% nulos) y `SkinThickness` (29.6% nulos); "
             "se imputó por mediana y se pasó de 768 → **761 registros**.")
    st.markdown("</div>", unsafe_allow_html=True)
    for img, cap in [
        ("m1_correlaciones.png", "Matriz de correlaciones"),
        ("m2_correlacion_objetivo.png", "Correlación con la variable objetivo"),
        ("m3_arbol.png", "Árbol de decisión (max_depth=3, var. dominante: Glucosa 0.713)"),
        ("m4_importancia_arbol.png", "Importancia de variables — Árbol / XGBoost"),
        ("m8_accuracy.png", "Accuracy por método"),
        ("m9_division.png", "División estratificada 70/30 + CV 10 pliegues"),
    ]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(cap)
        st.image(str(BASE / "figuras" / img), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB DATOS ----------------
with tab_datos:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Datos limpios (muestra)")
    st.dataframe(pd.read_csv(BASE / "datos_limpios.csv").head(10),
                 use_container_width=True)
    st.caption("761 filas × 7 columnas · 0 nulos · clases 64.9% / 35.1%.")
    st.subheader("🔮 5 casos futuros del informe")
    st.dataframe(pd.read_csv(BASE / "predicciones_futuras.csv"),
                 use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Metodología")
    st.markdown(
        "- **Objetivo:** clasificar riesgo de diabetes (Outcome 0/1).\n"
        "- **Limpieza:** ceros imposibles → NaN; drop Insulin/SkinThickness; "
        "imputación por mediana; 7 filas eliminadas.\n"
        "- **División:** 70/30 estratificada (532/229) + GridSearchCV con "
        "StratifiedKFold(10) sobre el train.\n"
        "- **Pipelines:** Tree/RF con discretización; KNN/SVM/MLP/XGB con MinMaxScaler "
        "(sin fuga de información).\n"
        "- **Overfitting/underfitting:** se controla con CV de 10 pliegues y test reservado; "
        "el árbol limitado a profundidad 3 y XGBoost con 50 árboles generalizan mejor "
        "que la red MLP (0.655, subajuste) sin memorizar el train.\n"
        "- **Hiperparámetros (GridSearch):** Tree max_depth=3 · MLP (32,8) · KNN k=5 · "
        "SVM RBF C=0.1 · RF 150 árboles · **XGB 50 árboles**.\n"
        "- **Despliegue:** este Streamlit carga `modelo_Xgboost.pkl` (scaler + clasificador).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="small">Proyecto minería predictiva · Parcial 1 · '
            "Modelo XGBoost serializado · Interfaz con fines académicos, no es diagnóstico médico.</p>",
            unsafe_allow_html=True)
