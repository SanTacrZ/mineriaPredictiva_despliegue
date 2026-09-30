"""Predictor de Diabetes (Pima Indians) — modelo final hiperparametrizado.

Ejecutar con el entorno del proyecto: ./env/bin/streamlit run app.py
Carga `modelo_final.pkl` (generado en el notebook, sección C). Si no existe,
usa `modelo_Xgboost.pkl`. Las métricas se leen de `resultados_medidas.csv`.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
FIG = BASE / "figuras"
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
# (mínimo, máximo, valor por defecto, paso)
RANGES = {
    "Pregnancies": (0, 17, 2, 1),
    "Glucose": (40, 250, 120, 1),
    "BloodPressure": (30, 130, 72, 1),
    "BMI": (15.0, 60.0, 32.3, 0.1),
    "DiabetesPedigreeFunction": (0.05, 2.50, 0.45, 0.01),
    "Age": (18, 85, 32, 1),
}
MEDIANAS = {"Glucose": 117.0, "BloodPressure": 72.0, "BMI": 32.3}
COLS_METRICAS = {"accuracy": "Accuracy", "precision": "Precisión", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC"}

st.set_page_config(
    page_title="Predictor de Diabetes",
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


# ---------------------------------------------------------------- carga
@st.cache_resource
def cargar_modelo():
    import pickle
    for nombre in ("modelo_final.pkl", "modelo_Xgboost.pkl"):
        ruta = BASE / nombre
        if ruta.exists():
            with open(ruta, "rb") as f:
                return pickle.load(f), nombre
    raise FileNotFoundError("No se encontró modelo_final.pkl ni modelo_Xgboost.pkl")


@st.cache_data
def cargar_metricas():
    tabla = pd.read_csv(BASE / "resultados_medidas.csv", index_col=0)
    tabla = tabla.rename(columns=COLS_METRICAS)  # por si vienen en minúscula
    return tabla


@st.cache_data
def matriz_confusion_test(nombre_modelo: str):
    """Reproduce el split 70/30 estratificado (random_state=42) del notebook
    sobre datos_limpios.csv y devuelve la matriz de confusión del modelo en test."""
    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(BASE / "datos_limpios.csv")
        X, y = df[FEATURES], df["Outcome"].astype(int)
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        modelo, _ = cargar_modelo()
        return confusion_matrix(y_te, modelo.predict(X_te)), len(X_te)
    except Exception:
        return None, 0


def importancias(modelo):
    """Importancia de variables si el último paso del pipeline la expone."""
    try:
        est = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
        imp = np.asarray(est.feature_importances_, dtype=float)
        if len(imp) == len(FEATURES):
            return dict(zip(FEATURES, imp))
    except Exception:
        pass
    return None


def descripcion_modelo(modelo):
    if hasattr(modelo, "steps"):
        pasos = " + ".join(type(p).__name__ for _, p in modelo.steps)
        return pasos
    return type(modelo).__name__


def predecir(df: pd.DataFrame, umbral: float):
    modelo, _ = cargar_modelo()
    proba = modelo.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= umbral).astype(int)
    return proba, pred


def gauge(prob: float, umbral: float):
    color = "#147A3E" if prob < umbral else "#B42318"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(prob * 100),
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probabilidad de diabetes", "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "steps": [{"range": [0, umbral * 100], "color": "#E6F7ED"},
                         {"range": [umbral * 100, 100], "color": "#FDECEC"}],
               "threshold": {"line": {"color": "#0B2545", "width": 3},
                             "value": umbral * 100}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def mostrar_img(nombre: str, titulo: str, caption: str = ""):
    ruta = FIG / nombre
    if ruta.exists():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(titulo)
        st.image(str(ruta), width="stretch")
        if caption:
            st.caption(caption)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- datos base
modelo, archivo_modelo = cargar_modelo()
tabla = cargar_metricas()
fila_final = "Modelo final" if "Modelo final" in tabla.index else tabla["Accuracy"].idxmax()
m = tabla.loc[fila_final]
imp = importancias(modelo)
desc_modelo = descripcion_modelo(modelo)

# ---------------------------------------------------------------- HERO
st.markdown(
    f"""<div class="hero">
    <h1>🩺 Predictor de Riesgo de Diabetes</h1>
    <p>Minería predictiva en Python · Dataset Pima Indians Diabetes (NIDDK) ·
    Modelo final: <b>{desc_modelo}</b> · validación cruzada 10 pliegues + GridSearch</p>
    <div class="chips"><span class="chip">🎯 Accuracy {m['Accuracy']:.3f}</span>
    <span class="chip">📈 ROC-AUC {m['ROC-AUC']:.3f}</span>
    <span class="chip">🔁 Recall {m['Recall']:.3f}</span>
    <span class="chip">🧪 6 variables clínicas</span></div></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.header("📝 Datos del paciente")
    st.caption("Ajusta las 6 variables clínicas: el resultado se actualiza solo.")
    vals = {}
    for feat in FEATURES:
        lo, hi, dflt, step = RANGES[feat]
        vals[feat] = st.slider(LABELS_ES[feat], min_value=lo, max_value=hi,
                               value=dflt, step=step)
    st.divider()
    umbral = st.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05)
    st.caption(f"Umbral actual: **{umbral:.2f}** (por defecto 0.50). "
               "Bajarlo aumenta la detección de casos (recall) a costa de más falsos positivos.")
    st.divider()
    st.caption(f"Pipeline: `{desc_modelo}` · archivo `{archivo_modelo}` · "
               f"Imputación por mediana: Glucosa {MEDIANAS['Glucose']:.0f} · "
               f"Presión {MEDIANAS['BloodPressure']:.0f} · IMC {MEDIANAS['BMI']}")

tab_pred, tab_mod, tab_evid, tab_datos = st.tabs(
    ["🔮 Predicción", "📊 Modelos comparados", "🖼️ Evidencia", "📁 Datos y metodología"])

# ---------------------------------------------------------------- TAB PREDICCIÓN
with tab_pred:
    col1, col2 = st.columns([1.05, 1])
    entrada = pd.DataFrame([vals], columns=FEATURES)
    proba, pred = predecir(entrada, umbral)
    p = float(proba[0])
    positivo = p >= umbral

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resultado")
        st.plotly_chart(gauge(p, umbral), width="stretch")
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
        elif p >= umbral:
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
                elif lote[FEATURES].isna().any().any():
                    st.error("El archivo tiene valores vacíos en las variables del modelo.")
                else:
                    pb, pr = predecir(lote, umbral)
                    out = lote[FEATURES].copy()
                    out["P(diabetes)"] = np.round(pb, 4)
                    out["Predicción"] = np.where(pr == 1, "DIABETES", "NO DIABETES")
                    st.dataframe(out, width="stretch")
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
                            "Valor": [vals[f] for f in FEATURES]})
        if imp:
            det["Importancia del modelo"] = [round(imp[f], 3) for f in FEATURES]
        st.dataframe(det, width="stretch", hide_index=True)
        if imp:
            st.bar_chart(det.set_index("Variable")["Importancia del modelo"])
            top = sorted(imp, key=imp.get, reverse=True)[:3]
            st.caption("Variables de mayor importancia en el modelo: "
                       + ", ".join(f"**{LABELS_ES[t]}** ({imp[t]:.3f})" for t in top) + ".")
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
        for t in msgs:
            st.write(t)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- TAB MODELOS
with tab_mod:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏆 Comparativa (test 30% · GridSearchCV 10 pliegues)")
    st.dataframe(tabla.style.format("{:.3f}").background_gradient(cmap="YlGn"),
                 width="stretch")
    mejor_auc = tabla["ROC-AUC"].idxmax()
    mejor_rec = tabla["Recall"].idxmax()
    st.success(f"**Modelo desplegado: {fila_final}** — Accuracy {m['Accuracy']:.3f} · "
               f"F1 {m['F1']:.3f} · Recall {m['Recall']:.3f} · ROC-AUC {m['ROC-AUC']:.3f}. "
               f"Mayor AUC de la tabla: {mejor_auc} ({tabla.loc[mejor_auc, 'ROC-AUC']:.3f}); "
               f"mayor recall: {mejor_rec} ({tabla.loc[mejor_rec, 'Recall']:.3f}).")
    st.bar_chart(tabla["Accuracy"])
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cm, n_te = matriz_confusion_test(archivo_modelo)
        cap = ""
        if cm is not None:
            cap = (f"Modelo desplegado en el test ({n_te} registros): "
                   f"[[{cm[0, 0]}, {cm[0, 1]}], [{cm[1, 0]}, {cm[1, 1]}]] "
                   "(filas = real, columnas = predicho).")
        mostrar_img("m6_confusion.png", "Matrices de confusión", cap)
    with c2:
        mostrar_img("m7_roc.png", "Curvas ROC")

# ---------------------------------------------------------------- TAB EVIDENCIA
with tab_evid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔎 Selección de factores y evidencias del notebook")
    st.write("**Correlación con Outcome:** Glucosa 0.491 · IMC 0.313 · Edad 0.235 · "
             "Embarazos 0.219 · DPF 0.172 · Presión 0.166. Ningún par supera |r|≥0.8 "
             "(máx. Embarazos–Edad ≈0.54): no hay redundancia grave. "
             "Se eliminaron `Insulin` (48.7% nulos) y `SkinThickness` (29.6% nulos); "
             "se imputó por mediana y se pasó de 768 → **761 registros**. "
             "Además se contrastó con información mutua, ANOVA e importancia por permutación.")
    st.markdown("</div>", unsafe_allow_html=True)

    for img, cap in [
        ("m1_correlaciones.png", "Matriz de correlaciones"),
        ("m2_correlacion_objetivo.png", "Correlación con la variable objetivo"),
        ("seleccion_factores.png", "Selección de factores (varios criterios)"),
        ("m3_arbol.png", "Árbol de decisión"),
        ("m4_importancia_arbol.png", "Importancia de variables — Árbol / XGBoost"),
        ("overfitting.png", "Revisión de overfitting: train vs CV vs test"),
        ("m8_accuracy.png", "Accuracy por método"),
        ("m9_division.png", "División estratificada 70/30 + CV 10 pliegues"),
    ]:
        mostrar_img(img, cap)

# ---------------------------------------------------------------- TAB DATOS
with tab_datos:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Datos limpios (muestra)")
    st.dataframe(pd.read_csv(BASE / "datos_limpios.csv").head(10),
                 width="stretch")
    st.caption("761 filas × 7 columnas · 0 nulos · clases 64.9% / 35.1%.")
    st.subheader("🔮 5 casos futuros del informe")
    st.dataframe(pd.read_csv(BASE / "predicciones_futuras.csv"),
                 width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Metodología")
    st.markdown(
        "- **Objetivo:** clasificar riesgo de diabetes (Outcome 0/1).\n"
        "- **Selección de factores:** correlación, información mutua, ANOVA e importancia "
        "por permutación; se descartan Insulin y SkinThickness por exceso de nulos.\n"
        "- **Limpieza:** ceros imposibles → NaN; imputación por mediana; 7 filas eliminadas.\n"
        "- **División:** 70/30 estratificada (532/229) + GridSearchCV con "
        "StratifiedKFold(10) sobre el train.\n"
        "- **Pipelines:** Tree/RF con discretización; KNN/SVM/MLP/XGB con MinMaxScaler "
        "(sin fuga de información).\n"
        "- **Overfitting/underfitting:** se compara accuracy de entrenamiento, validación "
        "cruzada y test por modelo (brecha train-CV > 0.08 ⇒ overfitting); ver la figura "
        "`overfitting.png` en la pestaña Evidencia.\n"
        "- **Hiperparámetros (GridSearch):** primera ronda por método y segunda ronda, más "
        f"amplia, sobre el mejor modelo por CV. Modelo desplegado: `{desc_modelo}`.\n"
        f"- **Despliegue:** esta app Streamlit carga `{archivo_modelo}` (pipeline completo).")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="small">Proyecto minería predictiva · Parcial 1 · '
            "Interfaz con fines académicos, no es diagnóstico médico.</p>",
            unsafe_allow_html=True)
