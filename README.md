# 🎓 PE Extractor Predictivo - Riesgo de Deserción Beca 18

Aplicación web serverless para predecir el **riesgo de deserción de becarios
de PRONABEC Beca 18**, usando **Python + Streamlit + Supabase (PostgreSQL)**.

Proyecto académico — Universidad Peruana Los Andes — Caso 5: PRONABEC Beca 18.

## Arquitectura

```
mi-proyecto-ml-pnda/
├── .streamlit/
│   └── secrets.toml          # Credenciales de Supabase (NO subir a git con datos reales)
├── models/
│   └── modelo_entrenado.pkl  # Pipeline (StandardScaler + CART) ya entrenado
├── Beca18.csv                 # Dataset real de PRONABEC (Nota Promedio del Postulante)
├── app.py                     # App Streamlit (frontend + backend)
├── entrenar_modelo.py         # Script de entrenamiento del modelo
├── supabase_schema.sql        # SQL para crear la tabla de auditoría
├── requirements.txt
└── .gitignore
```

## Modelo (ML Backend)

- **Modelo ganador**: Árbol de Decisión (CART), `max_depth=5`.
- **Features**: `Promedio` (11.0 - 20.0), `Ingreso_Familiar` (S/. 930 - 6000).
- **Target**: `Estado_Beca` (0 = Alto Riesgo de Deserción, 1 = Retiene la Beca).
- **Preprocesamiento**: `StandardScaler()` empaquetado junto al modelo dentro
  de un `Pipeline` de scikit-learn, exportado con `joblib` como
  `models/modelo_entrenado.pkl`. Esto evita reescribir el escalado en `app.py`.

### Sobre los datos

El CSV público de PRONABEC (`Dataset_Nota_Promedio_del_Postulante_Beca_18.csv`,
283,324 filas) solo trae el **Promedio académico** del postulante y datos de
su institución/colegio (región, provincia, gestión, año de convocatoria).
**No incluye** `Ingreso_Familiar` ni `Estado_Beca` (si el becario retuvo o
perdió la beca), porque esa información es confidencial y PRONABEC no la
libera públicamente.

Por eso, igual que en el análisis previo en Google Colab, `entrenar_modelo.py`
inyecta esas dos columnas mediante **data mocking**:
- `Ingreso_Familiar`: distribución uniforme entre 930 y 6000 soles.
- `Estado_Beca`: variable objetivo simulada, con 75% de probabilidad de
  retención (supuesto documentado y ajustable en el propio script).

**Importante para la sustentación**: como estas dos columnas son simuladas
aleatoriamente, no existe una relación causal real entre `Promedio` /
`Ingreso_Familiar` y `Estado_Beca` en los datos de prueba. Las métricas de
Accuracy/Precisión reflejan esto — es un resultado esperado y debe explicarse
así, no como un error del pipeline. Si tu universidad o PRONABEC te otorga
acceso a datos reales de deserción de becarios, basta con reemplazar esas
dos columnas simuladas por los valores reales y todo el pipeline sigue
funcionando igual.

## Base de datos (Supabase)

Tabla `predicciones_log` (auditoría de cada predicción hecha desde la app):

| Columna               | Tipo                      |
|------------------------|---------------------------|
| id                     | bigint, autoincremental, PK |
| fecha                  | timestamp with time zone, default now() |
| inputs_usuario         | jsonb                      |
| resultado_prediccion   | text                       |

## Instalación paso a paso

### 1. Clonar / descomprimir el proyecto

```bash
unzip mi-proyecto-ml-pnda.zip
cd mi-proyecto-ml-pnda
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar Supabase

1. Crea una cuenta y un proyecto en [supabase.com](https://supabase.com) (plan gratuito).
2. Ve al **SQL Editor** de tu proyecto y ejecuta todo el contenido de
   `supabase_schema.sql`. Esto crea la tabla `predicciones_log` con Row Level
   Security y políticas de inserción/lectura pública.
3. Ve a **Project Settings → API** y copia tu `Project URL` y tu `anon public key`.
4. Abre `.streamlit/secrets.toml` y reemplaza los valores de ejemplo:

```toml
SUPABASE_URL = "https://tu-proyecto.supabase.co"
SUPABASE_KEY = "tu_anon_key_real"
```

> ⚠️ Este archivo ya está en `.gitignore` — nunca lo subas a un repositorio
> público con tus credenciales reales.

### 4. (Re)entrenar el modelo

El archivo `models/modelo_entrenado.pkl` ya viene generado y listo para usar.
Si quieres regenerarlo (por ejemplo, tras cambiar el dataset o los supuestos
de simulación):

```bash
python entrenar_modelo.py
```

Esto imprime la comparación de 3 algoritmos (KNN, CART, Random Forest) igual
que en la Fase 2 de tu notebook de Colab, y guarda el Pipeline ganador
(CART) en `models/modelo_entrenado.pkl`.

### 5. Ejecutar la aplicación localmente

```bash
streamlit run app.py
```

Se abrirá en `http://localhost:8501`. La interfaz:
1. Pide **Promedio** e **Ingreso Familiar**.
2. Al presionar **"Ejecutar Predicción"**, transforma los datos con el
   scaler del pipeline, predice 0/1, y muestra:
   - ✅ **"Condición Estable"** si predice 1.
   - ⚠️ **"Riesgo Alto"** si predice 0.
3. Guarda el registro en Supabase **de forma silenciosa** (si falla, no
   interrumpe al usuario — solo aparece un detalle técnico opcional).

## Despliegue en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub (verifica que `.gitignore`
   esté excluyendo `.streamlit/secrets.toml`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta tu repo,
   indicando `app.py` como archivo principal.
3. En **App settings → Secrets**, pega el mismo contenido de tu
   `secrets.toml` local (con tus credenciales reales de Supabase).
4. Despliega. Streamlit instalará automáticamente `requirements.txt`.

## Notas finales

- El dataset `Beca18.csv` incluido es el dataset público real de PRONABEC
  ("Nota Promedio del Postulante Beca 18"), sin modificar salvo limpieza
  básica de nulos y rango del `Promedio`.
- Si tu institución obtiene datos reales de ingreso familiar y estado de la
  beca, solo reemplaza esas columnas en `entrenar_modelo.py` (elimina el
  bloque de "Data mocking") y vuelve a ejecutar el entrenamiento.
- Antes de un uso real con datos de becarios reales, agrega autenticación de
  usuarios a la app — los datos socioeconómicos de estudiantes son sensibles
  (Ley de Protección de Datos Personales, Ley N° 29733, Perú).
