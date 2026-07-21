"""
Caso 5 - PRONABEC Beca 18: Predicción de Riesgo de Deserción
Universidad Peruana Los Andes

Frontend + Backend en Streamlit. Carga el modelo CART entrenado
(models/modelo_entrenado.pkl), predice el riesgo de deserción a partir del
Promedio académico y el Ingreso Familiar, y registra cada predicción en
Supabase (tabla predicciones_log) para auditoría.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from supabase import create_client

# -----------------------------------------------------------------
# Configuración de la página
# -----------------------------------------------------------------
st.set_page_config(
    page_title="PE Extractor Predictivo - Beca 18",
    page_icon="🎓",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "modelo_entrenado.pkl"


# -----------------------------------------------------------------
# Carga del modelo (cacheada: se carga una sola vez por sesión de servidor)
# -----------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    import subprocess
    # Forzar el reentrenamiento en la nube porque el código de simulación cambió
    print("Reentrenando modelo con nueva lógica realista...")
    subprocess.run(["python", "entrenar_modelo.py"], check=True)
    return joblib.load(MODEL_PATH)


# -----------------------------------------------------------------
# Cliente de Supabase (cacheado; credenciales desde st.secrets)
# -----------------------------------------------------------------
@st.cache_resource
def cargar_cliente_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def guardar_log_supabase(promedio: float, ingreso_familiar: float, resultado_texto: str):
    """Guarda el registro de auditoría en Supabase de forma silenciosa.
    Si falla (credenciales no configuradas, sin red, etc.), no interrumpe
    la experiencia del usuario: solo se registra el error en un expander.
    """
    try:
        client = cargar_cliente_supabase()
        registro = {
            "fecha": datetime.now(timezone.utc).isoformat(),
            "inputs_usuario": {
                "Promedio": promedio,
                "Ingreso_Familiar": ingreso_familiar,
            },
            "resultado_prediccion": resultado_texto,
        }
        client.table("predicciones_log").insert(registro).execute()
        return True, None
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------
st.title("PE Extractor Predictivo - Riesgo de Deserción Beca 18")
st.markdown(
    "Modelo: **Árbol de Decisión (CART, max_depth=5)** · "
    "Ingresa el promedio académico y el ingreso familiar del postulante/becario "
    "para estimar su riesgo de deserción de la beca."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    promedio = st.number_input(
        "Promedio académico",
        min_value=11.0,
        max_value=20.0,
        value=14.0,
        step=0.1,
        format="%.1f",
        help="Promedio académico del postulante/becario (escala 11.0 - 20.0).",
    )
with col2:
    ingreso_familiar = st.number_input(
        "Ingreso familiar mensual (S/.)",
        min_value=930.0,
        max_value=6000.0,
        value=1500.0,
        step=10.0,
        format="%.2f",
        help="Ingreso familiar mensual en soles.",
    )

ejecutar = st.button("Ejecutar Predicción", use_container_width=True, type="primary")

if ejecutar:
    modelo = cargar_modelo()

    # Construimos el DataFrame de entrada con los mismos nombres de columna
    # usados en el entrenamiento, para que el Pipeline (scaler + CART) lo
    # procese sin errores.
    entrada = pd.DataFrame([{
        "Promedio": promedio,
        "Ingreso_Familiar": ingreso_familiar,
    }])

    prediccion = int(modelo.predict(entrada)[0])
    proba = modelo.predict_proba(entrada)[0]

    if prediccion == 1:
        resultado_texto = "Condición Estable"
        st.success(f"✅ **Condición Estable** — probabilidad de retención: {proba[1]*100:.1f}%")
    else:
        resultado_texto = "Riesgo Alto"
        st.error(f"⚠️ **Riesgo Alto de Deserción** — probabilidad de riesgo: {proba[0]*100:.1f}%")

    # Guardado silencioso en Supabase (no bloquea ni distrae al usuario)
    ok, error = guardar_log_supabase(promedio, ingreso_familiar, resultado_texto)
    if not ok:
        with st.expander("Detalle técnico (auditoría no guardada)"):
            st.caption(f"No se pudo registrar en Supabase: {error}")

st.divider()
st.caption(
    "Caso 5 · PRONABEC Beca 18 · Universidad Peruana Los Andes · "
    "Proyecto académico — Promedio basado en datos reales de PRONABEC; "
    "Ingreso_Familiar y Estado_Beca son datos simulados (data mocking) "
    "por no estar disponibles públicamente."
)
