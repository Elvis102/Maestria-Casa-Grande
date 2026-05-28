# ============================================================
# app.py - Predictor de Rendimiento Académico Estudiantil
# Proyecto Final Integrador S3
# Maestría en Inteligencia Artificial y Ciencia de Datos
# Elvis Mauricio Sánchez Rogel - Universidad Casa Grande
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_CSV = Path(__file__).parent / "Student_performance_data_.csv"
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              precision_score, recall_score)

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Predictor de Rendimiento Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a3a6b;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #eef4fb;
        border-left: 4px solid #1a3a6b;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a3a6b;
        border-bottom: 2px solid #1a3a6b;
        padding-bottom: 4px;
        margin: 1.2rem 0 0.8rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a3a6b;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Mapeo de etiquetas ───────────────────────────────────────
GRADE_MAP = {
    0.0: "A - Excelente",
    1.0: "B - Bueno",
    2.0: "C - Regular",
    3.0: "D - Bajo",
    4.0: "F - Reprobado"
}
GRADE_COLORS = {
    "A - Excelente": "#27ae60",
    "B - Bueno":     "#2ecc71",
    "C - Regular":   "#f39c12",
    "D - Bajo":      "#e67e22",
    "F - Reprobado": "#e74c3c"
}
GENDER_MAP     = {0: "Femenino", 1: "Masculino"}
ETHNICITY_MAP  = {0: "Caucásico", 1: "Afroamericano", 2: "Asiático", 3: "Otro"}
PAREDU_MAP     = {0: "Sin educación formal", 1: "Bachillerato",
                  2: "Técnico / Superior", 3: "Pregrado", 4: "Posgrado"}
PARSUP_MAP     = {0: "Ninguno", 1: "Bajo", 2: "Moderado", 3: "Alto", 4: "Muy alto"}
BINARY_MAP     = {0: "No", 1: "Sí"}

FEATURE_LABELS = {
    "Age":               "Edad",
    "StudyTimeWeekly":   "Horas de estudio/semana",
    "Absences":          "Ausencias",
    "GPA":               "GPA",
    "ParentalEducation": "Educación parental",
    "ParentalSupport":   "Apoyo parental",
    "Tutoring":          "Recibe tutoría",
    "Extracurricular":   "Actividades extracurriculares",
}

# ── Carga de datos ───────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(DEFAULT_CSV)
    return df


@st.cache_data
def enrich_df(df):
    dfe = df.copy()
    dfe["Calificación"]         = dfe["GradeClass"].map(GRADE_MAP)
    dfe["Género"]               = dfe["Gender"].map(GENDER_MAP)
    dfe["Etnia"]                = dfe["Ethnicity"].map(ETHNICITY_MAP)
    dfe["Educación Parental"]   = dfe["ParentalEducation"].map(PAREDU_MAP)
    dfe["Apoyo Parental"]       = dfe["ParentalSupport"].map(PARSUP_MAP)
    dfe["Tutoría"]              = dfe["Tutoring"].map(BINARY_MAP)
    dfe["Extracurricular"]      = dfe["Extracurricular"].map(BINARY_MAP)
    dfe["En Riesgo"]            = (dfe["GradeClass"] >= 3).map({True: "En riesgo", False: "Sin riesgo"})
    return dfe


# ── Entrenamiento de modelos (cacheado) ──────────────────────
FEATURES = ["Age", "StudyTimeWeekly", "Absences", "GPA",
            "ParentalEducation", "ParentalSupport", "Tutoring", "Extracurricular"]


@st.cache_resource
def train_models(df):
    dm = df.dropna(subset=["GradeClass"])
    X  = dm[FEATURES]
    y  = (dm["GradeClass"] >= 3).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr  = LogisticRegression(max_iter=500, random_state=42)

    rf.fit(X_train,  y_train)
    gb.fit(X_train,  y_train)
    lr.fit(X_tr_s,   y_train)

    results = {}
    for name, model, X_t, X_v in [
        ("Random Forest",        rf,  X_train, X_test),
        ("Gradient Boosting",    gb,  X_train, X_test),
        ("Regresión Logística",  lr,  X_tr_s,  X_te_s),
    ]:
        y_pred = model.predict(X_v)
        results[name] = {
            "model":     model,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "accuracy":  round(accuracy_score(y_test, y_pred),  3),
            "f1":        round(f1_score(y_test, y_pred),         3),
            "precision": round(precision_score(y_test, y_pred),  3),
            "recall":    round(recall_score(y_test, y_pred),     3),
        }
    return results, scaler, rf


# ── Barra lateral ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Rendimiento Académico")
    st.divider()
    seccion = st.radio(
        "Navegación",
        ["🏠 Inicio",
         "📂 Exploración de Datos",
         "📊 Análisis Visual",
         "🤖 Modelo Predictivo",
         "📋 Conclusiones"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Dataset**")
    uploaded = st.file_uploader(
        "Cargar CSV personalizado (opcional)",
        type=["csv"],
        help="Si no cargas ningún archivo, se usa el dataset de Kaggle incluido por defecto."
    )
    if uploaded is None:
        st.success("✅ Dataset pre-cargado  \n`Student_performance_data_.csv`")
    else:
        st.info(f"📂 Usando: `{uploaded.name}`")
    st.divider()
    st.caption(
        "**Elvis M. Sánchez Rogel**  \n"
        "Maestría IA y Ciencia de Datos  \n"
        "Universidad Casa Grande · 2025"
    )

# Cargar y enriquecer datos
df  = load_data(uploaded)
dfe = enrich_df(df)

# ── SECCIÓN 1: Inicio ────────────────────────────────────────
if seccion == "🏠 Inicio":
    st.markdown('<p class="main-header">🎓 Predictor de Rendimiento Académico</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Análisis predictivo de factores que influyen en el desempeño estudiantil '
        'mediante Machine Learning — Universidad Casa Grande · 2025'
        '</p>',
        unsafe_allow_html=True
    )

    # KPIs
    total       = len(df)
    en_riesgo   = int((df["GradeClass"] >= 3).sum())
    pct_riesgo  = en_riesgo / total * 100
    gpa_prom    = df["GPA"].mean()
    gpa_riesgo  = df[df["GradeClass"] >= 3]["GPA"].mean()
    hrs_prom    = df["StudyTimeWeekly"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Estudiantes", f"{total:,}")
    c2.metric("Variables",   f"{len(FEATURES)}")
    c3.metric("En riesgo (D/F)", f"{en_riesgo:,}", f"{pct_riesgo:.1f}% del total")
    c4.metric("GPA promedio", f"{gpa_prom:.2f}")
    c5.metric("Hrs estudio/sem", f"{hrs_prom:.1f}")

    st.divider()

    col_txt, col_chart = st.columns([3, 2])

    with col_txt:
        st.markdown("#### Sobre este proyecto")
        st.markdown(
            "Esta aplicación analiza el *Student Performance Dataset* (Kaggle, 2023) "
            "para identificar los factores que predicen el rendimiento académico en "
            "estudiantes de secundaria. Integra exploración de datos, visualización "
            "interactiva y modelos de clasificación supervisada para generar una "
            "herramienta de apoyo a la toma de decisiones educativas.\n\n"
            "**Objetivos de la aplicación:**\n"
            "- Explorar la distribución de calificaciones y variables asociadas\n"
            "- Visualizar correlaciones y comparativas entre grupos de estudiantes\n"
            "- Entrenar modelos predictivos para identificar estudiantes en riesgo\n"
            "- Proveer un predictor interactivo para nuevos casos individuales\n\n"
            "**Dataset:** 2 392 estudiantes · 13 variables predictoras · "
            "1 variable objetivo (GradeClass)"
        )

    with col_chart:
        counts = dfe["Calificación"].value_counts().reset_index()
        counts.columns = ["Calificación", "Estudiantes"]
        fig_pie = px.pie(
            counts,
            values="Estudiantes",
            names="Calificación",
            title="Distribución de Calificaciones",
            color="Calificación",
            color_discrete_map=GRADE_COLORS,
            hole=0.42
        )
        fig_pie.update_layout(height=310, margin=dict(t=40, b=0, l=0, r=0),
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Tabla resumen del dataset
    st.markdown("#### Descripción de las variables")
    var_desc = pd.DataFrame({
        "Variable": ["Age", "Gender", "Ethnicity", "ParentalEducation",
                     "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
                     "Extracurricular", "Sports", "Music", "Volunteering", "GPA", "GradeClass"],
        "Descripción": [
            "Edad del estudiante (15–18)",
            "Género (0=Femenino, 1=Masculino)",
            "Etnia (0=Caucásico, 1=Afroamericano, 2=Asiático, 3=Otro)",
            "Nivel educativo de los padres (0–4)",
            "Horas de estudio semanales",
            "Número de ausencias escolares",
            "Si recibe tutoría académica (0/1)",
            "Nivel de apoyo parental (0–4)",
            "Actividades extracurriculares (0/1)",
            "Practica deportes (0/1)",
            "Toca algún instrumento musical (0/1)",
            "Participa en voluntariado (0/1)",
            "Promedio académico acumulado (0.0–4.0)",
            "Clase de calificación final (0=A … 4=F)"
        ],
        "Tipo": ["Numérica", "Categórica", "Categórica", "Categórica",
                 "Numérica", "Numérica", "Categórica", "Categórica",
                 "Categórica", "Categórica", "Categórica", "Categórica",
                 "Numérica", "Objetivo"]
    })
    st.dataframe(var_desc, use_container_width=True, hide_index=True)


# ── SECCIÓN 2: Exploración de datos ─────────────────────────
elif seccion == "📂 Exploración de Datos":
    st.markdown("## 📂 Exploración de Datos")

    tab_prev, tab_stats, tab_missing = st.tabs(
        ["Vista previa", "Estadísticas descriptivas", "Calidad del dato"]
    )

    # ---- Tab: Vista previa ----
    with tab_prev:
        st.markdown("#### Vista previa del dataset original")
        n = st.slider("Número de filas", 5, 100, 10)
        st.dataframe(df.head(n), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", f"{df.shape[0]:,}")
        col2.metric("Columnas", df.shape[1])
        col3.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.markdown("#### Tipos de datos")
        dtype_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo":    df.dtypes.astype(str).values
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True,
                     column_config={"Tipo": st.column_config.TextColumn(width="small")})

    # ---- Tab: Estadísticas descriptivas ----
    with tab_stats:
        st.markdown("#### Estadísticas de variables numéricas")
        vars_num = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
        desc = df[vars_num].describe().round(3).T
        desc.index = ["Edad", "Horas estudio/sem", "Ausencias", "GPA"]
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### Distribución del GPA")
        fig_gpa = px.histogram(
            df, x="GPA", nbins=40,
            title="Distribución del GPA (Promedio Académico)",
            color_discrete_sequence=["#2980b9"],
            labels={"GPA": "GPA", "count": "Frecuencia"}
        )
        fig_gpa.add_vline(x=df["GPA"].mean(), line_dash="dash",
                          line_color="red",
                          annotation_text=f"Media: {df['GPA'].mean():.2f}",
                          annotation_position="top right")
        fig_gpa.update_layout(height=320)
        st.plotly_chart(fig_gpa, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribución de ausencias")
            fig_abs = px.histogram(
                df, x="Absences", nbins=30,
                title="Histograma de Ausencias",
                color_discrete_sequence=["#e74c3c"],
                labels={"Absences": "Ausencias", "count": "Frecuencia"}
            )
            fig_abs.update_layout(height=280)
            st.plotly_chart(fig_abs, use_container_width=True)

        with col2:
            st.markdown("#### Horas de estudio semanales")
            fig_hrs = px.histogram(
                df, x="StudyTimeWeekly", nbins=25,
                title="Histograma de Horas de Estudio",
                color_discrete_sequence=["#27ae60"],
                labels={"StudyTimeWeekly": "Horas/Semana", "count": "Frecuencia"}
            )
            fig_hrs.update_layout(height=280)
            st.plotly_chart(fig_hrs, use_container_width=True)

    # ---- Tab: Calidad del dato ----
    with tab_missing:
        st.markdown("#### Análisis de valores faltantes")
        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df  = pd.DataFrame({
            "Columna":    missing.index,
            "Faltantes":  missing.values,
            "% Faltantes": missing_pct.values
        })

        if missing.sum() == 0:
            st.success("✅ El dataset no presenta valores faltantes en ninguna columna.")
        else:
            st.warning(f"⚠️ Se encontraron {missing.sum()} valores faltantes.")

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        st.markdown("#### Duplicados")
        dupes = df.duplicated().sum()
        if dupes == 0:
            st.success("✅ No se encontraron filas duplicadas.")
        else:
            st.warning(f"⚠️ Se encontraron {dupes} filas duplicadas.")

        st.markdown("#### Rango de valores clave")
        range_df = pd.DataFrame({
            "Variable": ["Age", "GPA", "StudyTimeWeekly", "Absences"],
            "Mínimo":   [df["Age"].min(), df["GPA"].min(),
                         df["StudyTimeWeekly"].min(), df["Absences"].min()],
            "Máximo":   [df["Age"].max(), df["GPA"].max(),
                         df["StudyTimeWeekly"].max(), df["Absences"].max()],
            "Media":    [round(df["Age"].mean(), 2), round(df["GPA"].mean(), 2),
                         round(df["StudyTimeWeekly"].mean(), 2),
                         round(df["Absences"].mean(), 2)],
        })
        st.dataframe(range_df, use_container_width=True, hide_index=True)


# ── SECCIÓN 3: Análisis Visual ───────────────────────────────
elif seccion == "📊 Análisis Visual":
    st.markdown("## 📊 Análisis Visual Exploratorio")

    tab_dist, tab_corr, tab_comp = st.tabs(
        ["Distribuciones", "Correlaciones", "Análisis comparativo"]
    )

    # ---- Tab: Distribuciones ----
    with tab_dist:
        col1, col2 = st.columns(2)

        with col1:
            counts = dfe["Calificación"].value_counts().reset_index()
            counts.columns = ["Calificación", "Estudiantes"]
            fig_bar = px.bar(
                counts, x="Calificación", y="Estudiantes",
                title="Distribución de Calificaciones Finales",
                color="Calificación",
                color_discrete_map=GRADE_COLORS
            )
            fig_bar.update_layout(height=350, showlegend=False,
                                  xaxis_tickangle=-15)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_abs_box = px.box(
                dfe, x="Calificación", y="Absences",
                title="Ausencias por Nivel de Calificación",
                color="Calificación",
                color_discrete_map=GRADE_COLORS,
                labels={"Absences": "Número de ausencias"}
            )
            fig_abs_box.update_layout(height=350, showlegend=False,
                                      xaxis_tickangle=-15)
            st.plotly_chart(fig_abs_box, use_container_width=True)

        # Scatter GPA vs Study time
        fig_scat = px.scatter(
            dfe, x="StudyTimeWeekly", y="GPA",
            color="Calificación",
            color_discrete_map=GRADE_COLORS,
            title="Horas de Estudio vs GPA por Nivel de Calificación",
            labels={"StudyTimeWeekly": "Horas de estudio / semana", "GPA": "GPA"},
            opacity=0.65, size_max=6
        )
        fig_scat.update_layout(height=380)
        st.plotly_chart(fig_scat, use_container_width=True)

    # ---- Tab: Correlaciones ----
    with tab_corr:
        vars_corr = ["Age", "StudyTimeWeekly", "Absences", "GPA",
                     "ParentalEducation", "ParentalSupport",
                     "Tutoring", "Extracurricular", "GradeClass"]
        labels_corr = ["Edad", "Hrs estudio", "Ausencias", "GPA",
                       "Educ. parental", "Apoyo parental",
                       "Tutoría", "Extracurricular", "GradeClass"]

        corr_m = df[vars_corr].corr()
        corr_m.index   = labels_corr
        corr_m.columns = labels_corr

        fig_heat = px.imshow(
            corr_m.round(2),
            text_auto=True,
            title="Matriz de Correlación de Pearson",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_heat.update_layout(height=520)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown(
            '<div class="info-box">'
            '💡 <b>Interpretación:</b> GPA tiene correlación negativa fuerte con GradeClass '
            '(−0.92 aprox.), lo que confirma que GPA alto equivale a calificación baja en número '
            '(0 = A). Ausencias también tiene correlación positiva con GradeClass (más ausencias → '
            'calificación numérica más alta → peor desempeño). Horas de estudio muestran correlación '
            'negativa moderada con GradeClass.'
            '</div>',
            unsafe_allow_html=True
        )

    # ---- Tab: Análisis comparativo ----
    with tab_comp:
        col1, col2 = st.columns(2)

        with col1:
            # GPA por género
            fig_vio = px.violin(
                dfe, x="Género", y="GPA",
                color="Género",
                box=True, points="outliers",
                title="Distribución del GPA por Género",
                color_discrete_sequence=["#e74c3c", "#3498db"]
            )
            fig_vio.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)

        with col2:
            # GPA promedio por educación parental
            gpa_edu = (dfe.groupby("Educación Parental")["GPA"]
                         .mean().reset_index()
                         .rename(columns={"GPA": "GPA Promedio"}))
            gpa_edu["GPA Promedio"] = gpa_edu["GPA Promedio"].round(3)
            fig_edu = px.bar(
                gpa_edu.sort_values("GPA Promedio"),
                x="GPA Promedio", y="Educación Parental",
                orientation="h",
                title="GPA Promedio por Nivel de Educación Parental",
                color="GPA Promedio",
                color_continuous_scale="Greens"
            )
            fig_edu.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_edu, use_container_width=True)

        # Riesgo por apoyo parental
        risk_sup = (dfe.groupby("Apoyo Parental")["En Riesgo"]
                      .apply(lambda x: (x == "En riesgo").mean() * 100)
                      .reset_index()
                      .rename(columns={"En Riesgo": "% En Riesgo"}))
        fig_risk = px.bar(
            risk_sup,
            x="Apoyo Parental", y="% En Riesgo",
            title="Porcentaje de Estudiantes en Riesgo según Apoyo Parental",
            color="% En Riesgo",
            color_continuous_scale="Reds",
            labels={"% En Riesgo": "% en riesgo (D o F)"}
        )
        fig_risk.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

        # Tutoría vs riesgo
        col3, col4 = st.columns(2)
        with col3:
            tut_risk = (dfe.groupby("Tutoría")["En Riesgo"]
                          .apply(lambda x: (x == "En riesgo").mean() * 100)
                          .reset_index()
                          .rename(columns={"En Riesgo": "% En Riesgo"}))
            fig_tut = px.bar(
                tut_risk, x="Tutoría", y="% En Riesgo",
                title="Riesgo según Tutoría",
                color="Tutoría",
                color_discrete_sequence=["#95a5a6", "#27ae60"]
            )
            fig_tut.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_tut, use_container_width=True)

        with col4:
            ext_risk = (dfe.groupby("Extracurricular")["En Riesgo"]
                          .apply(lambda x: (x == "En riesgo").mean() * 100)
                          .reset_index()
                          .rename(columns={"En Riesgo": "% En Riesgo"}))
            fig_ext = px.bar(
                ext_risk, x="Extracurricular", y="% En Riesgo",
                title="Riesgo según Extracurriculares",
                color="Extracurricular",
                color_discrete_sequence=["#95a5a6", "#3498db"]
            )
            fig_ext.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_ext, use_container_width=True)


# ── SECCIÓN 4: Modelo Predictivo ─────────────────────────────
elif seccion == "🤖 Modelo Predictivo":
    st.markdown("## 🤖 Modelo Predictivo de Riesgo Académico")

    results, scaler, rf_model = train_models(df)

    tab_metrics, tab_predict = st.tabs(
        ["Entrenamiento y métricas", "Predicción individual"]
    )

    # ---- Tab: Métricas ----
    with tab_metrics:
        st.markdown(
            '<div class="info-box">'
            '🎯 <b>Objetivo del modelo:</b> clasificar si un estudiante está <b>en riesgo '
            'académico</b> (GradeClass D o F) o no, a partir de 8 variables predictoras. '
            'Se entrenaron 3 algoritmos con validación 80/20.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Tabla comparativa de modelos
        metrics_df = pd.DataFrame([
            {
                "Modelo":     name,
                "Accuracy":   res["accuracy"],
                "F1-Score":   res["f1"],
                "Precision":  res["precision"],
                "Recall":     res["recall"]
            }
            for name, res in results.items()
        ])
        st.markdown("#### Comparación de modelos")
        st.dataframe(
            metrics_df.set_index("Modelo").style.highlight_max(
                axis=0, color="#d5f5e3"
            ),
            use_container_width=True
        )

        st.markdown("#### Detalle por modelo")
        model_sel = st.selectbox(
            "Selecciona modelo",
            list(results.keys()),
            label_visibility="collapsed"
        )
        res = results[model_sel]

        col1, col2 = st.columns(2)

        with col1:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{res['accuracy']:.3f}")
            c1.metric("Precision", f"{res['precision']:.3f}")
            c2.metric("F1-Score",  f"{res['f1']:.3f}")
            c2.metric("Recall",    f"{res['recall']:.3f}")

            # Confusion matrix
            cm = confusion_matrix(res["y_test"], res["y_pred"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                title=f"Matriz de Confusión — {model_sel}",
                labels=dict(x="Predicho", y="Real", color="N"),
                x=["Sin riesgo", "En riesgo"],
                y=["Sin riesgo", "En riesgo"],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=320)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            if model_sel in ("Random Forest", "Gradient Boosting"):
                model_obj = res["model"]
                fi = pd.DataFrame({
                    "Variable":    [FEATURE_LABELS[f] for f in FEATURES],
                    "Importancia": model_obj.feature_importances_
                }).sort_values("Importancia")
                fig_fi = px.bar(
                    fi, x="Importancia", y="Variable",
                    orientation="h",
                    title=f"Importancia de Variables — {model_sel}",
                    color="Importancia",
                    color_continuous_scale="Blues"
                )
                fig_fi.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                model_obj = res["model"]
                coef = pd.DataFrame({
                    "Variable": [FEATURE_LABELS[f] for f in FEATURES],
                    "Coeficiente": np.abs(model_obj.coef_[0])
                }).sort_values("Coeficiente")
                fig_coef = px.bar(
                    coef, x="Coeficiente", y="Variable",
                    orientation="h",
                    title="Coeficientes absolutos — Regresión Logística",
                    color="Coeficiente",
                    color_continuous_scale="Purples"
                )
                fig_coef.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_coef, use_container_width=True)

    # ---- Tab: Predicción individual ----
    with tab_predict:
        st.markdown("#### Ingresar datos del estudiante")
        st.markdown("Completa el perfil del estudiante para obtener la predicción de riesgo académico.")

        col1, col2, col3 = st.columns(3)

        with col1:
            age_p    = st.slider("Edad", 15, 18, 17)
            study_p  = st.slider("Horas de estudio / semana", 0.0, 20.0, 10.0, 0.5)
            absent_p = st.slider("Número de ausencias", 0, 30, 5)

        with col2:
            gpa_p    = st.slider("GPA (0.0 – 4.0)", 0.0, 4.0, 2.5, 0.1)
            paredu_p = st.selectbox("Educación parental",
                                    options=list(PAREDU_MAP.keys()),
                                    format_func=lambda x: PAREDU_MAP[x])
            parsup_p = st.selectbox("Apoyo parental",
                                    options=list(PARSUP_MAP.keys()),
                                    format_func=lambda x: PARSUP_MAP[x])

        with col3:
            tutor_p = st.selectbox("¿Recibe tutoría?",
                                   [0, 1], format_func=lambda x: BINARY_MAP[x])
            extra_p = st.selectbox("¿Actividades extracurriculares?",
                                   [0, 1], format_func=lambda x: BINARY_MAP[x])
            st.markdown("")
            predict_btn = st.button("🔮 Predecir riesgo académico",
                                    type="primary", use_container_width=True)

        if predict_btn:
            X_new   = np.array([[age_p, study_p, absent_p, gpa_p,
                                  paredu_p, parsup_p, tutor_p, extra_p]])
            pred    = rf_model.predict(X_new)[0]
            proba   = rf_model.predict_proba(X_new)[0]
            riesgo  = proba[1] * 100
            seguro  = proba[0] * 100

            st.divider()

            if pred == 1:
                st.error(
                    f"⚠️ **ESTUDIANTE EN RIESGO ACADÉMICO**  \n"
                    f"Probabilidad de riesgo: **{riesgo:.1f}%**"
                )
                st.markdown("""
                **Intervenciones recomendadas:**
                - 📚 Incrementar horas de estudio semanales hacia el rango 10–15 h
                - 🏫 Reducir ausencias: cada día faltado impacta significativamente
                - 👨‍🏫 Inscribirse en programa de tutoría académica
                - 👨‍👩‍👧 Involucrar a la familia en el seguimiento académico
                - 📋 Diseñar un plan de estudio personalizado con el tutor
                """)
            else:
                st.success(
                    f"✅ **ESTUDIANTE SIN RIESGO ACADÉMICO**  \n"
                    f"Probabilidad de riesgo: **{riesgo:.1f}%**"
                )
                st.markdown(
                    "El perfil del estudiante indica buenos indicadores académicos. "
                    "Se recomienda mantener los hábitos actuales de estudio y asistencia."
                )

            # Gauge de probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=riesgo,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": "Probabilidad de Riesgo Académico"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e74c3c" if pred == 1 else "#27ae60"},
                    "steps": [
                        {"range": [0,  30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fef9e7"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "value": 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)


# ── SECCIÓN 5: Conclusiones ──────────────────────────────────
elif seccion == "📋 Conclusiones":
    st.markdown("## 📋 Conclusiones y Hallazgos")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Hallazgos principales del análisis")
        st.markdown("""
        1. **El GPA es el predictor dominante** del nivel de calificación final: la
           correlación entre ambas variables supera 0.92, lo que indica que el promedio
           acumulado refleja fielmente el desempeño del estudiante.

        2. **Las ausencias son el segundo factor de impacto**: los estudiantes con
           calificación F presentan en promedio más del doble de ausencias que los
           estudiantes con calificación A, confirmando que la asistencia es una
           variable crítica en el rendimiento.

        3. **El apoyo parental tiene efecto protector**: a mayor nivel de apoyo familiar,
           menor es el porcentaje de estudiantes en riesgo. Estudiantes con apoyo
           "Muy alto" muestran una tasa de riesgo significativamente menor.

        4. **La tutoría reduce el riesgo**: los estudiantes que reciben tutoría tienen
           menor proporción de calificaciones D y F, lo que valida las intervenciones
           pedagógicas de acompañamiento.

        5. **El Random Forest superó a los demás modelos**, alcanzando el mayor F1-score
           y accuracy en la tarea de clasificación binaria (en riesgo / sin riesgo),
           lo que confirma la utilidad de los métodos de ensemble para este tipo de
           problema educativo.

        6. **El 67.9% de los estudiantes** del dataset obtiene calificación D o F,
           lo que evidencia una problemática académica sistémica que justifica el
           uso de modelos predictivos para intervención temprana.
        """)

        st.markdown("### Propuesta de aplicación institucional")
        st.markdown("""
        Esta herramienta puede integrarse en los sistemas de gestión académica de
        universidades e instituciones de secundaria para ejecutar un **análisis
        de riesgo al inicio de cada período lectivo**. Con los datos de historial
        académico, asistencia y contexto familiar, el modelo identifica con alta
        precisión qué estudiantes requieren intervención antes de que su rendimiento
        se deteriore irreversiblemente.

        El flujo propuesto integra las cuatro herramientas analizadas en el módulo:
        **Scikit-learn** para el modelado, **PyCaret** para la comparación y optimización
        de modelos, **LazyPredict** como exploración inicial de algoritmos, y
        **Streamlit** para la presentación interactiva a coordinadores y docentes.
        """)

    with col_right:
        # Gauge resumen
        en_riesgo_pct = (df["GradeClass"] >= 3).mean() * 100
        fig_gauge2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=en_riesgo_pct,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Estudiantes en Riesgo (D/F)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c"},
                "steps": [
                    {"range": [0,  30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": 50
                }
            }
        ))
        fig_gauge2.update_layout(height=270)
        st.plotly_chart(fig_gauge2, use_container_width=True)

        st.markdown("### Variables de mayor impacto")
        impact_df = pd.DataFrame({
            "Variable": ["GPA", "Ausencias", "Horas de estudio",
                         "Apoyo parental", "Tutoría"],
            "Impacto": [5, 4, 3, 3, 2]
        }).sort_values("Impacto", ascending=True)
        fig_impact = px.bar(
            impact_df, x="Impacto", y="Variable",
            orientation="h",
            color="Impacto",
            color_continuous_scale="Blues",
            title="Importancia relativa (escala 1–5)"
        )
        fig_impact.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig_impact, use_container_width=True)

    st.divider()
    st.markdown("### Referencias")
    st.markdown("""
    - Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
      *Journal of Machine Learning Research, 12*, 2825–2830.
    - Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow*
      (2nd ed.). O'Reilly Media.
    - Streamlit Inc. (2023). *Streamlit documentation*. https://docs.streamlit.io
    - Kaggle. (2023). *Students Performance Dataset*.
      https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
    - Müller, A. C., & Guido, S. (2016). *Introduction to machine learning with Python*.
      O'Reilly Media.
    """)
