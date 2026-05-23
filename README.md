# 🎓 Predictor de Rendimiento Académico Estudiantil

**Proyecto Final Integrador S3 — Maestría en Inteligencia Artificial y Ciencia de Datos**  
**Autor:** Elvis Mauricio Sánchez Rogel · Universidad Casa Grande · 2025

---

## ¿Qué hace esta aplicación?

`app.py` es una aplicación interactiva construida con **Streamlit** que permite cargar, explorar, visualizar y predecir el rendimiento académico de estudiantes de secundaria. Utiliza técnicas de clasificación supervisada (Machine Learning) para identificar estudiantes en riesgo de obtener calificaciones bajas (D o F), a partir de variables académicas, familiares y conductuales.

La aplicación está orientada a coordinadores académicos y docentes que necesiten una herramienta de **apoyo a la toma de decisiones** sin requerir conocimientos de programación.

---

## Dataset

| Atributo | Valor |
|---|---|
| **Nombre** | Student Performance Dataset |
| **Fuente** | Kaggle — Rabie El Kharoua (2023) |
| **Filas** | 2 392 estudiantes |
| **Columnas** | 15 variables |
| **Archivo** | `Student_performance_data_.csv` |

### Variables del dataset

| Variable | Descripción | Tipo |
|---|---|---|
| `StudentID` | Identificador único del estudiante | Numérica |
| `Age` | Edad (15 a 18 años) | Numérica |
| `Gender` | Género — 0: Femenino, 1: Masculino | Categórica |
| `Ethnicity` | Etnia — 0: Caucásico, 1: Afroamericano, 2: Asiático, 3: Otro | Categórica |
| `ParentalEducation` | Nivel educativo de los padres (0 = sin educación formal … 4 = posgrado) | Categórica |
| `StudyTimeWeekly` | Horas de estudio por semana | Numérica |
| `Absences` | Número de ausencias escolares | Numérica |
| `Tutoring` | Si el estudiante recibe tutoría — 0: No, 1: Sí | Categórica |
| `ParentalSupport` | Nivel de apoyo parental (0 = ninguno … 4 = muy alto) | Categórica |
| `Extracurricular` | Participa en actividades extracurriculares — 0/1 | Categórica |
| `Sports` | Practica deportes — 0/1 | Categórica |
| `Music` | Toca instrumento musical — 0/1 | Categórica |
| `Volunteering` | Participa en voluntariado — 0/1 | Categórica |
| `GPA` | Promedio académico acumulado (0.0 a 4.0) | Numérica |
| `GradeClass` | **Variable objetivo** — Clase de calificación: 0=A, 1=B, 2=C, 3=D, 4=F | Objetivo |

> El 67.9 % de los estudiantes del dataset obtiene calificación D o F, lo que justifica el análisis predictivo para intervención temprana.

---

## Estructura del proyecto

```
S3-Proyecto_final_integrador/
├── app.py                          # Aplicación Streamlit principal
├── Student_performance_data_.csv   # Dataset de Kaggle
├── requirements.txt                # Dependencias Python
├── Informe_Proyecto_Final_Sanchez.md  # Informe académico formal
└── README.md                       # Este archivo
```

---

## Estructura de la aplicación (`app.py`)

La aplicación se organiza en **5 secciones** navegables desde la barra lateral:

### 🏠 1. Inicio
Panel de bienvenida con:
- **5 KPIs en tiempo real:** total de estudiantes, variables analizadas, cantidad y porcentaje en riesgo, GPA promedio, horas de estudio promedio.
- Gráfico de torta (donut) con la distribución de calificaciones A–F.
- Tabla descriptiva de todas las variables del dataset.

### 📂 2. Exploración de Datos
Tres pestañas internas:

| Pestaña | Contenido |
|---|---|
| **Vista previa** | Tabla interactiva con control de filas (5–100), tipos de datos, dimensiones y uso de memoria |
| **Estadísticas descriptivas** | Tabla con media, desviación estándar, min/max para Age, StudyTimeWeekly, Absences, GPA. Histogramas con línea de la media para GPA, Ausencias y Horas de estudio |
| **Calidad del dato** | Detección de valores faltantes por columna, filas duplicadas, tabla de rangos válidos |

### 📊 3. Análisis Visual
Tres pestañas con gráficos interactivos (Plotly):

| Pestaña | Gráficos |
|---|---|
| **Distribuciones** | Barras de calificaciones por nivel, boxplot de ausencias por calificación, scatter de horas de estudio vs GPA coloreado por calificación |
| **Correlaciones** | Mapa de calor (heatmap) de la matriz de correlación de Pearson entre 9 variables, con anotaciones de valores |
| **Análisis comparativo** | Violin plot GPA por género, barras de GPA promedio por educación parental, barras de % en riesgo según apoyo parental, tutoría y actividades extracurriculares |

### 🤖 4. Modelo Predictivo
Dos pestañas:

**Entrenamiento y métricas:**
- Entrena 3 modelos de clasificación binaria (en riesgo = GradeClass ≥ 3 vs. sin riesgo):

| Modelo | Algoritmo | Parámetros clave |
|---|---|---|
| Random Forest | `RandomForestClassifier` | 200 árboles, random_state=42 |
| Gradient Boosting | `GradientBoostingClassifier` | 100 estimadores, random_state=42 |
| Regresión Logística | `LogisticRegression` | max_iter=500, datos estandarizados |

- División: 80 % entrenamiento / 20 % prueba con estratificación.
- Tabla comparativa de Accuracy, F1-Score, Precision y Recall con resaltado del mejor valor.
- Selector interactivo de modelo para ver: métricas individuales, matriz de confusión e importancia de variables (o coeficientes absolutos para Regresión Logística).

**Predicción individual:**
- Formulario con 8 controles (sliders + selectboxes) para ingresar el perfil de un estudiante.
- Botón "Predecir riesgo académico" que ejecuta el Random Forest en tiempo real.
- Resultado: alerta verde (sin riesgo) o roja (en riesgo) con probabilidad exacta.
- Medidor gauge animado con la probabilidad de riesgo.
- Si hay riesgo: lista de intervenciones pedagógicas recomendadas.

### 📋 5. Conclusiones
- 6 hallazgos clave del análisis con sus evidencias cuantitativas.
- Propuesta de integración institucional del flujo completo (LazyPredict → Scikit-learn → PyCaret → Streamlit).
- Gauge resumen del porcentaje global de estudiantes en riesgo.
- Gráfico de importancia relativa de las 5 variables más influyentes.
- Referencias bibliográficas en formato APA 7.ª edición.

---

## Variables usadas para el modelo

De las 15 columnas del dataset, se seleccionaron **8 variables predictoras** y **1 variable objetivo binaria**:

```python
FEATURES = [
    "Age",               # Edad
    "StudyTimeWeekly",   # Horas de estudio / semana
    "Absences",          # Número de ausencias
    "GPA",               # Promedio académico
    "ParentalEducation", # Nivel educativo de los padres
    "ParentalSupport",   # Apoyo parental
    "Tutoring",          # Recibe tutoría
    "Extracurricular"    # Actividades extracurriculares
]

# Variable objetivo (binaria)
y = (GradeClass >= 3)   # 1 = En riesgo (D o F), 0 = Sin riesgo (A, B o C)
```

---

## Resultados del modelo

| Modelo | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| **Random Forest** | **0.927** | **0.948** | **0.935** | **0.961** |
| Gradient Boosting | 0.918 | 0.941 | 0.928 | 0.955 |
| Regresión Logística | 0.708 | 0.786 | 0.763 | 0.811 |

El **Random Forest** con 200 estimadores obtuvo el mejor desempeño en todas las métricas.

---

## Cómo correr la aplicación

### Opción A — Entorno virtual (recomendado localmente)

```bash
# 1. Clonar o descargar el proyecto
cd S3-Proyecto_final_integrador

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

### Opción B — Con el venv ya creado (si ya lo instalaste)

```bash
cd S3-Proyecto_final_integrador
.venv/bin/streamlit run app.py
```

La app abre en `http://localhost:8501`

### Opción C — Streamlit Cloud (acceso público)

La aplicación está desplegada en:  
**[maestria-casa-grande.streamlit.app](https://maestria-casa-grande-qclmmzcb25hmugt2o77pou.streamlit.app/)**

---

## Dependencias

```
streamlit>=1.28.0
pandas>=1.5.0
numpy<2
plotly>=5.15.0
scikit-learn>=1.3.0
```

> Se requiere `numpy<2` por incompatibilidad de varios paquetes del ecosistema Anaconda con NumPy 2.x.

---

## Decisiones técnicas del código

| Mecanismo | Propósito |
|---|---|
| `@st.cache_data` | Cachea la carga del CSV y el enriquecimiento del dataframe para no releer el archivo en cada interacción |
| `@st.cache_resource` | Cachea el entrenamiento de los 3 modelos para que solo se ejecuten una vez por sesión |
| `stratify=y` en `train_test_split` | Mantiene la proporción de clases en entrenamiento y prueba, evitando sesgo por desbalance |
| `StandardScaler` | Estandariza las variables solo para Regresión Logística (los árboles no lo requieren) |
| `st.file_uploader` | Permite subir un CSV propio; si no se sube, carga el dataset por defecto |
| Plotly Express | Gráficos interactivos con tooltips, zoom y descarga de imagen integrados nativamente en Streamlit |

---

## Referencias

- El Kharoua, R. (2023). *Students Performance Dataset*. Kaggle. https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Streamlit Inc. (2023). *Streamlit documentation*. https://docs.streamlit.io
- Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
