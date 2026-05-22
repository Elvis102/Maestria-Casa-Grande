# Predictor de Rendimiento Académico Estudiantil: Aplicación en Streamlit con Dataset de Kaggle

**Autor:** Elvis Mauricio Sánchez Rogel  
**Programa:** Maestría en Inteligencia Artificial y Ciencia de Datos  
**Institución:** Universidad Casa Grande  
**Fecha:** Mayo 2025  

---

## Título

**Predictor de Rendimiento Académico Estudiantil: Análisis Exploratorio y Clasificación Supervisada Mediante Streamlit**

---

## Introducción

El rendimiento académico estudiantil constituye una de las problemáticas centrales en las instituciones de educación secundaria y superior a nivel global. La detección temprana de estudiantes en situación de riesgo académico —definida como aquellos con alta probabilidad de obtener calificaciones bajas o de abandonar sus estudios— permite a las instituciones diseñar estrategias de intervención pedagógica oportunas y focalizadas, optimizando sus recursos de tutoría y orientación.

La convergencia entre la disponibilidad de datos educativos estructurados, el avance en herramientas de aprendizaje automático de código abierto y la capacidad de construir aplicaciones interactivas sin infraestructura compleja ha creado una oportunidad sin precedentes para llevar el análisis predictivo al ámbito de la gestión académica.

El presente proyecto integra los conocimientos adquiridos durante el módulo de Paradigmas en Inteligencia Artificial y Ciencia de Datos para desarrollar una aplicación funcional en Streamlit que analiza el *Student Performance Dataset* (Kaggle, 2023), con el propósito de explorar, visualizar y modelar el rendimiento académico estudiantil mediante técnicas de clasificación supervisada.

---

## Problemática

Las instituciones educativas enfrentan el desafío de identificar, con la debida anticipación, qué estudiantes presentan mayor riesgo de fracaso académico. Los enfoques tradicionales de seguimiento —revisiones periódicas de calificaciones, reuniones con tutores, reportes de asistencia— resultan reactivos: actúan después de que el deterioro del rendimiento se ha manifestado.

El análisis de datos educativos mediante modelos de machine learning ofrece una alternativa proactiva: a partir de variables como el historial de asistencia, las horas de estudio semanales, el nivel de apoyo familiar y el promedio académico acumulado, es posible predecir con alta precisión qué estudiantes tienen mayor probabilidad de obtener calificaciones D o F antes de que el período lectivo concluya.

Sin embargo, estos modelos predictivos resultan de escasa utilidad si no son accesibles para el personal académico no técnico —coordinadores, docentes, orientadores— que debe tomar las decisiones de intervención. Streamlit resuelve esta brecha al convertir un modelo de machine learning en una aplicación web interactiva, consultable por cualquier usuario sin conocimientos de programación.

---

## Objetivos

### Objetivo general

Desarrollar una aplicación interactiva en Streamlit que permita la carga, exploración, análisis visual y modelado predictivo del rendimiento académico estudiantil, integrando técnicas de clasificación supervisada con una interfaz accesible para usuarios no técnicos.

### Objetivos específicos

1. Explorar el *Student Performance Dataset* mediante estadísticas descriptivas y análisis de calidad del dato para comprender la distribución de las variables.
2. Visualizar las relaciones entre variables académicas, familiares y conductuales mediante gráficos interactivos que permitan identificar patrones relevantes.
3. Entrenar y evaluar tres modelos de clasificación supervisada (Random Forest, Gradient Boosting y Regresión Logística) para la predicción binaria de riesgo académico.
4. Implementar un módulo de predicción individual que permita ingresar el perfil de un estudiante y obtener una estimación de su riesgo académico en tiempo real.
5. Desplegar la aplicación en Streamlit Cloud para garantizar accesibilidad pública sin instalación local.

---

## Justificación

Este proyecto se justifica en múltiples dimensiones:

**Relevancia social:** El 67.6% de los estudiantes del dataset analizado obtiene una calificación D o F, evidenciando una problemática sistémica que afecta la equidad educativa y los resultados institucionales. Herramientas de detección temprana pueden revertir esta tendencia con intervenciones oportunas.

**Pertinencia profesional:** El autor se desempeña en el ámbito educativo y tiene interés académico en la aplicación de inteligencia artificial para la mejora del éxito estudiantil, lo que hace de este proyecto un ejercicio directamente transferible a su contexto profesional.

**Integración de competencias:** El proyecto consolida el uso de Python para ciencia de datos (pandas, Scikit-learn, Plotly), el paradigma de desarrollo de aplicaciones reactivas con Streamlit, y los principios de validación y evaluación de modelos de clasificación aprendidos durante el módulo.

**Valor demostrativo:** La aplicación sirve como evidencia concreta de que los modelos de machine learning pueden ser accesibles para personal no técnico, cerrando la brecha entre el análisis predictivo y la toma de decisiones institucional.

---

## Marco Teórico

### Ciencia de Datos en el contexto educativo

La Ciencia de Datos Educativos (*Educational Data Mining*, EDM) es una disciplina emergente que aplica técnicas de análisis de datos y aprendizaje automático al estudio de fenómenos educativos (Baker & Yacef, 2009). Sus aplicaciones incluyen la predicción del abandono estudiantil, la personalización de contenidos de aprendizaje y la detección de patrones de comportamiento académico.

### Clasificación supervisada

La clasificación supervisada es una categoría del aprendizaje automático en la que un modelo aprende a asignar una etiqueta de clase a nuevas observaciones, a partir del entrenamiento sobre un conjunto de datos etiquetado (James et al., 2023). Para el presente proyecto se implementaron tres algoritmos:

- **Random Forest:** ensemble de árboles de decisión que promedia las predicciones de múltiples árboles entrenados sobre submuestras del dataset, reduciendo la varianza y mejorando la generalización (Breiman, 2001).
- **Gradient Boosting:** ensemble secuencial que construye árboles de decisión en forma iterativa, donde cada árbol corrige los errores del anterior, logrando alta precisión en problemas con relaciones no lineales (Friedman, 2001).
- **Regresión Logística:** modelo lineal probabilístico que estima la probabilidad de pertenencia a una clase mediante una función sigmoide, utilizado como línea base por su interpretabilidad (Müller & Guido, 2016).

### Métricas de evaluación

Para la evaluación de los modelos se emplearon las siguientes métricas estándar para clasificación binaria (Pedregosa et al., 2011):

- **Accuracy:** proporción de predicciones correctas sobre el total.
- **F1-Score:** media armónica de Precision y Recall, robusta ante clases desbalanceadas.
- **Precision:** proporción de verdaderos positivos sobre el total de predicciones positivas.
- **Recall:** proporción de verdaderos positivos detectados sobre el total real de positivos.

### Streamlit como plataforma de despliegue

Streamlit es un framework de Python que permite construir aplicaciones web interactivas para ciencia de datos sin requerir conocimientos de HTML, CSS o JavaScript (Streamlit Inc., 2023). Su modelo de programación reactiva re-ejecuta el script completo ante cada interacción del usuario, lo que simplifica el manejo de estado y la actualización dinámica de visualizaciones.

---

## Metodología

### Dataset utilizado

El proyecto utiliza el *Student Performance Dataset* publicado en Kaggle por Rabie El Kharoua (2023), disponible en: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

El dataset contiene registros de 2 392 estudiantes de secundaria con las siguientes características:

| Variable | Descripción | Tipo |
|---|---|---|
| StudentID | Identificador único | Numérica |
| Age | Edad (15–18 años) | Numérica |
| Gender | Género (0=Femenino, 1=Masculino) | Categórica |
| Ethnicity | Etnia (0–3) | Categórica |
| ParentalEducation | Nivel educativo de los padres (0–4) | Categórica |
| StudyTimeWeekly | Horas de estudio semanales | Numérica |
| Absences | Número de ausencias escolares | Numérica |
| Tutoring | Recibe tutoría académica (0/1) | Categórica |
| ParentalSupport | Nivel de apoyo parental (0–4) | Categórica |
| Extracurricular | Actividades extracurriculares (0/1) | Categórica |
| Sports | Practica deportes (0/1) | Categórica |
| Music | Toca instrumento musical (0/1) | Categórica |
| Volunteering | Participa en voluntariado (0/1) | Categórica |
| GPA | Promedio académico acumulado (0.0–4.0) | Numérica |
| GradeClass | Clase de calificación final (0=A … 4=F) | Objetivo |

### Preprocesamiento

El dataset no presentó valores faltantes ni filas duplicadas, lo que eliminó la necesidad de imputación. Para el entrenamiento de modelos lineales (Regresión Logística) se aplicó estandarización con `StandardScaler`. La variable objetivo se transformó en binaria: **en riesgo** (GradeClass ≥ 3, equivalente a D o F) vs. **sin riesgo** (GradeClass ≤ 2).

La división del dataset siguió la proporción estándar 80/20 para entrenamiento y prueba, con estratificación para mantener la distribución de clases en ambos conjuntos.

### Variables predictoras

Se seleccionaron 8 variables como predictoras, excluyendo `StudentID` (identificador sin valor predictivo) y `Ethnicity`, `Sports`, `Music`, `Volunteering` (baja correlación con la variable objetivo):

- Age, StudyTimeWeekly, Absences, GPA
- ParentalEducation, ParentalSupport, Tutoring, Extracurricular

---

## Desarrollo de la Aplicación

### Arquitectura del proyecto

```
S3-Proyecto final integrador/
├── app.py                        # Aplicación principal Streamlit
├── requirements.txt              # Dependencias
├── Student_performance_data_.csv # Dataset de Kaggle
├── .streamlit/
│   └── config.toml              # Tema visual personalizado
└── Informe_Proyecto_Final_Sanchez.md
```

### Estructura de la aplicación

La aplicación se organizó en cinco secciones accesibles desde la barra lateral:

**1. Inicio:** Panel de bienvenida con cinco métricas clave (total de estudiantes, variables analizadas, porcentaje en riesgo, GPA promedio, horas de estudio promedio), descripción del proyecto, tabla de variables y gráfico de distribución de calificaciones.

**2. Exploración de Datos:** Tres pestañas internas: vista previa del dataset con control deslizante para número de filas; estadísticas descriptivas de variables numéricas con histogramas; y análisis de calidad del dato (valores faltantes, duplicados, rangos válidos).

**3. Análisis Visual:** Distribuciones con histogramas y boxplots interactivos; matriz de correlación de Pearson con anotaciones; y análisis comparativo por género, educación parental, apoyo familiar, tutoría y actividades extracurriculares.

**4. Modelo Predictivo:** Pestaña de entrenamiento con comparación de tres modelos (Random Forest, Gradient Boosting, Regresión Logística), métricas detalladas, matriz de confusión e importancia de variables; y pestaña de predicción individual con formulario interactivo y medidor gauge de probabilidad de riesgo.

**5. Conclusiones:** Resumen de hallazgos, propuesta de aplicación institucional, medidor gauge del porcentaje general de riesgo y referencias bibliográficas.

### Decisiones técnicas relevantes

- Se utilizó `@st.cache_data` para la carga del dataset y `@st.cache_resource` para el entrenamiento de modelos, evitando recomputaciones en cada interacción del usuario.
- El modelo Random Forest se configuró con 200 estimadores y semilla fija (random_state=42) para garantizar reproducibilidad de resultados.
- Se empleó Plotly Express en lugar de Matplotlib para todas las visualizaciones, aprovechando la interactividad nativa con Streamlit (tooltips, zoom, descarga de imagen).
- El tema visual se personalizó mediante `.streamlit/config.toml` con una paleta de colores institucional (azul oscuro primario).

---

## Resultados

### Análisis exploratorio

El análisis exploratorio reveló los siguientes patrones significativos:

**Distribución de calificaciones:** El 50.6% de los estudiantes obtuvo calificación F (reprobado) y el 17.3% calificación D, totalizando el **67.9% en riesgo académico**. Solo el 4.5% alcanzó calificación A.

**Correlaciones:** La matriz de correlación mostró que GPA tiene la correlación negativa más fuerte con GradeClass (≈ −0.92), confirmando que el promedio acumulado es el predictor más potente. Las ausencias presentaron correlación positiva moderada con GradeClass (≈ 0.58), y las horas de estudio correlación negativa moderada (≈ −0.18).

**Apoyo parental:** Los estudiantes con nivel de apoyo parental "Ninguno" presentaron la mayor tasa de riesgo (>75%), mientras que los de apoyo "Muy alto" mostraron tasas inferiores al 55%.

**Tutoría:** Los estudiantes que reciben tutoría tienen menor proporción de calificaciones D/F, validando la efectividad de las intervenciones pedagógicas de acompañamiento.

### Resultados del modelado

| Modelo | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.927 | 0.948 | 0.935 | 0.961 |
| Gradient Boosting | 0.918 | 0.941 | 0.928 | 0.955 |
| Regresión Logística | 0.708 | 0.786 | 0.763 | 0.811 |

*Tabla de resultados. Clasificación binaria: en riesgo (GradeClass ≥ 3) vs. sin riesgo.*

El **Random Forest** obtuvo el mejor desempeño con un F1-Score de 0.948, seguido por Gradient Boosting con 0.941. La Regresión Logística, si bien útil como línea base interpretable, fue significativamente superada por los métodos de ensemble.

Las variables de mayor importancia según el Random Forest fueron, en orden descendente: GPA (≈60% de importancia), Absences (≈20%), StudyTimeWeekly (≈8%), ParentalSupport (≈5%) y Tutoring (≈3%).

---

## Conclusiones

1. El GPA constituye el predictor dominante del rendimiento académico, con una importancia relativa superior al 60% en el modelo Random Forest, lo que subraya la necesidad de dar seguimiento continuo al promedio acumulado del estudiante.

2. Las ausencias son el segundo factor de mayor impacto: los estudiantes en riesgo presentan el doble de ausencias que los estudiantes sin riesgo, confirmando que la asistencia escolar es una variable crítica y modificable mediante intervención institucional.

3. El apoyo parental y la tutoría académica actúan como factores protectores significativos, lo que justifica políticas institucionales que promuevan la participación familiar y los programas de acompañamiento estudiantil.

4. El Random Forest alcanzó un F1-Score de 0.948 en la clasificación de riesgo académico, lo que demuestra la viabilidad de modelos de ensemble para esta aplicación en contextos educativos reales.

5. Streamlit permitió transformar el análisis predictivo en una aplicación accesible para usuarios no técnicos, cerrando la brecha entre el modelado estadístico y la toma de decisiones pedagógica.

6. El proyecto evidencia que la integración de Scikit-learn (modelado), Plotly (visualización) y Streamlit (despliegue) constituye un flujo completo y eficiente para proyectos de ciencia de datos educativos de bajo a mediano volumen.

---

## Referencias

Baker, R. S., & Yacef, K. (2009). The state of educational data mining in 2009: A review and future visions. *Journal of Educational Data Mining, 1*(1), 3–17.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An introduction to statistical learning with applications in Python*. Springer.

Kaggle / El Kharoua, R. (2023). *Students Performance Dataset*. Kaggle. https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

Müller, A. C., & Guido, S. (2016). *Introduction to machine learning with Python: A guide for data scientists*. O'Reilly Media.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Streamlit Inc. (2023). *Streamlit documentation: The fastest way to build and share data apps*. https://docs.streamlit.io

VanderPlas, J. (2016). *Python data science handbook: Essential tools for working with data*. O'Reilly Media.
