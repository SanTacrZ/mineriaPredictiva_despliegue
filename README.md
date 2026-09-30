# 🩺 Predictor de Riesgo de Diabetes — mineriaPredictiva_despliegue

Despliegue del mejor modelo del Parcial 1 de Minería Predictiva (Pima Indians Diabetes, NIDDK).

- **Modelo:** `Pipeline(MinMaxScaler + XGBClassifier n_estimators=50)` — `modelo_Xgboost.pkl`
- **Validación:** GridSearchCV con StratifiedKFold(10) sobre train 70% + test 30% reservado
- **Calidad (test):** Accuracy **0.769** · Precisión 0.696 · Recall **0.600** · F1 0.644 · ROC-AUC 0.807
- **App:** Streamlit profesional (paleta azul marino + verde teal), predicción individual y por lote (CSV), comparativa de 6 modelos y evidencias.

## ▶️ Ejecución local (con el `env` del proyecto)

```bash
cd mineriaPredictiva_despliegue
./env/bin/streamlit run app.py
# o activando el entorno:
source env/bin/activate
streamlit run app.py
```

> El entorno `env/` ya incluye: streamlit 1.64.0 · xgboost 3.4.1 · scikit-learn 1.9.1 · pandas 3.0.6 · numpy 2.5.3 · plotly 7.1.0.
> Si lo recreas desde cero: `python3 -m venv env && ./env/bin/pip install -r requirements.txt`

## ☁️ Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub (ver comandos abajo).
2. En [share.streamlit.io](https://share.streamlit.io) → *New app* → repositorio `SanTacrZ/mineriaPredictiva_despliegue`, rama `main`, archivo `app.py`.
3. Deploy. Adjuntar el pantallazo de la app corriendo (punto D del parcial).

## 📁 Estructura

```
app.py                    # interfaz Streamlit
modelo_Xgboost.pkl        # pipeline ganador serializado
results.json / resultados_medidas.csv  # métricas de los 6 modelos
datos_limpios.csv         # 761 registros limpios
predicciones_futuras.csv  # 5 casos futuros del informe
figuras/                  # m1..m9 (correlaciones, árbol, confusión, ROC, accuracy, división)
requirements.txt          # dependencias pineadas (mismas del env)
.streamlit/config.toml    # tema profesional
```

## 🧪 Variables del modelo

| # | Variable | Entrada app |
|---|----------|-------------|
| 1 | Pregnancies | Embarazos (0–17) |
| 2 | Glucose | Glucosa mg/dL (40–250) |
| 3 | BloodPressure | Presión mm Hg (30–130) |
| 4 | BMI | IMC (15–60) |
| 5 | DiabetesPedigreeFunction | DPF (0.05–2.50) |
| 6 | Age | Edad (18–85) |

Umbral de decisión: 0.5 (ajustable en la barra lateral).

⚠️ Uso académico: apoyo didáctico, no constituye diagnóstico médico.
