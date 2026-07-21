"""
Caso 5 - PRONABEC Beca 18: Entrenamiento del modelo de riesgo de deserción.

Modelo ganador (Fase 2 del análisis en Colab): Árbol de Decisión (CART),
max_depth=5. Este script entrena un Pipeline (StandardScaler + CART) sobre
el dataset real de PRONABEC (Nota Promedio del Postulante Beca 18) y lo
exporta como models/modelo_entrenado.pkl con joblib.

NOTA SOBRE LOS DATOS: el CSV público de PRONABEC ("Nota Promedio del
Postulante Beca 18") solo trae el Promedio académico del postulante y datos
de la institución/colegio; NO incluye Ingreso_Familiar ni Estado_Beca
(si el becario retuvo o perdió la beca), porque esa información es
confidencial. Por eso se inyectan (data mocking) las columnas:
  - Ingreso_Familiar: uniforme entre 930 y 6000 soles.
  - Estado_Beca: variable objetivo simulada (0=Alto riesgo, 1=Retiene),
    con probabilidades condicionadas al Promedio y al Ingreso_Familiar
    para generar resultados realistas y correlacionados.

Se usa un Pipeline de scikit-learn (StandardScaler + DecisionTreeClassifier)
para que el escalado quede empaquetado junto con el modelo y no haya que
re-implementarlo en app.py.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Beca18.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------
# 1. Cargar el dataset real de PRONABEC
# -----------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False, sep=",", on_bad_lines="skip")

df["Promedio"] = pd.to_numeric(df["Promedio"], errors="coerce")
df["Promedio"] = df["Promedio"].fillna(df["Promedio"].median())
# El dataset real trae algunos promedios fuera de rango típico (0-20); recortamos
# al rango académico válido acordado (11.0 - 20.0, nota mínima aprobatoria).
df["Promedio"] = df["Promedio"].clip(lower=11.0, upper=20.0)

# -----------------------------------------------------------------
# 2. Data mocking: simulación del contexto socioeconómico faltante
#    (Ingreso_Familiar y Estado_Beca no existen en el CSV público)
# -----------------------------------------------------------------
num_filas = len(df)
df["Ingreso_Familiar"] = np.random.uniform(930, 6000, num_filas)

# Simulación realista condicionada
prob_retencion = np.full(num_filas, 0.85) # Base 85% de retención
prob_retencion = np.where(df["Promedio"] < 13.0, prob_retencion - 0.50, prob_retencion)
prob_retencion = np.where(df["Promedio"] >= 16.0, prob_retencion + 0.10, prob_retencion)
prob_retencion = np.where(df["Ingreso_Familiar"] < 1500, prob_retencion - 0.25, prob_retencion)
prob_retencion = np.clip(prob_retencion, 0.05, 0.98) # Limitar entre 5% y 98%

# Asignar 0 o 1 basado en la probabilidad individual
random_vals = np.random.rand(num_filas)
df["Estado_Beca"] = (random_vals < prob_retencion).astype(int)

FEATURES = ["Promedio", "Ingreso_Familiar"]
TARGET = "Estado_Beca"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------------------------------------------
# 3. Comparación de algoritmos (réplica de la Fase 2 de tu Colab),
#    cada uno como Pipeline (StandardScaler + modelo)
# -----------------------------------------------------------------
modelos = {
    "K-Vecinos Cercanos (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Árbol de Decisión (CART)": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
}

resultados = []
pipelines = {}

for nombre, modelo in modelos.items():
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", modelo),
    ])
    pipeline.fit(X_train, y_train)
    pipelines[nombre] = pipeline

    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    end_time = time.time()

    resultados.append({
        "Algoritmo": nombre,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precisión": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Tiempo Inferencia (s)": round(end_time - start_time, 6),
    })

print("\n" + "=" * 75)
print(" RESULTADOS COMPARATIVOS")
print("=" * 75)
print(pd.DataFrame(resultados).to_string(index=False))
print("=" * 75)

# -----------------------------------------------------------------
# 4. Exportar el modelo ganador: Árbol de Decisión (CART)
# -----------------------------------------------------------------
MODELO_GANADOR = "Árbol de Decisión (CART)"
pipeline_final = pipelines[MODELO_GANADOR]

modelo_path = MODELS_DIR / "modelo_entrenado.pkl"
joblib.dump(pipeline_final, modelo_path)

print(f"\n✅ Modelo ganador exportado: {MODELO_GANADOR}")
print(f"   Guardado en: {modelo_path}")
print(f"   Features: {FEATURES}")
print(f"   Target: {TARGET} (0=Alto Riesgo de Deserción, 1=Retiene la Beca)")
