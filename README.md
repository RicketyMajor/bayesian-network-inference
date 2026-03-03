# Bayesian Network Inference & Data Augmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![pgmpy](https://img.shields.io/badge/Library-pgmpy-orange.svg)
![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-E95420.svg)
![Machine Learning](https://img.shields.io/badge/Field-Probabilistic%20AI-success.svg)

## Descripción del Proyecto

Este proyecto implementa un ciclo de vida completo de Machine Learning utilizando **Redes Bayesianas (Modelos Gráficos Probabilísticos)**. A diferencia de los modelos de caja negra tradicionales, este proyecto se enfoca en la **interpretabilidad, el razonamiento causal y el diagnóstico inverso**.

Utilizando el clásico _Adult Census Income Dataset_ de la UCI, el objetivo es descubrir la estructura causal subyacente de los datos demográficos, parametrizar la red para inferir la probabilidad de que una persona gane `>50K` al año, y finalmente, utilizar la IA generativa para aumentar el conjunto de datos mediante datos sintéticos.

---

## Arquitectura y Fases del Proyecto

1. **Análisis Exploratorio y Discretización (`notebooks/`):** Tratamiento de datos nulos y discretización justificada de variables continuas (como la edad) para la correcta asimilación en Tablas de Probabilidad Condicional (CPDs).
2. **Aprendizaje de Estructura (`src/structure_learning.py`):** * **Score-based:** Búsqueda *Hill-Climbing\* utilizando el criterio de información bayesiano (BIC).
   - **Constraint-based:** _Algoritmo PC_ utilizando pruebas de independencia condicional (Chi-cuadrado).
3. **Estimación de Parámetros e Inferencia (`src/inference.py`):** Estimación por Máxima Verosimilitud (MLE) e inferencia exacta utilizando el algoritmo _Variable Elimination_.
4. **Data Augmentation Generativo (`src/synthetic_gen.py`):** Uso de _Forward Sampling_ para generar un 50% de datos adicionales puramente sintéticos basados en la distribución conjunta aprendida.
5. **Evaluación Comparativa (`src/evaluate_augmented.py`):** Contraste del rendimiento del modelo base vs. el modelo entrenado con el Súper-Dataset.

---

## Resultados y Análisis Teórico

### 1. Grafo Causal Descubierto (Hill-Climbing)

_(Nota: Las imágenes de los grafos generados se encuentran en `results/figures/dag_hill_climbing.png`)_.
El modelo descubrió relaciones probabilísticas fuertes, identificando que variables como `education` y `age` son ancestros causales directos del nivel de ingresos (`income`).

### 2. Inferencias Diagnósticas (Razonamiento Inverso)

A diferencia de un árbol de decisión, esta red bayesiana permitió realizar inferencias inversas. Por ejemplo, al consultar la probabilidad del estado civil dado que _ya sabemos_ que la persona gana más de 50K (`P(Marital-Status | Income=>50K)`), el modelo dedujo correctamente un 85.06% de probabilidad de que la persona esté casada (`Married-civ-spouse`), demostrando un fuerte entendimiento sociológico del censo.

### 3. La Paradoja del Data Augmentation

| Métrica (Test Set 30%)          | Baseline (70% Datos Reales) | Augmented (Reales + 50% Sintéticos) |
| :------------------------------ | :-------------------------: | :---------------------------------: |
| **Exactitud Global (Accuracy)** |           69.10%            |               69.00%                |
| **F1-Score (Clase >50K)**       |            0.26             |                0.27                 |

**Conclusión Teórica:** El _Accuracy_ se mantuvo estático debido a la naturaleza de los modelos generativos entrenados por MLE. Al generar datos sintéticos mediante _Forward Sampling_, la red replicó con éxito la distribución y **los sesgos (desbalance de clases)** del dataset original. Esto demuestra que inyectar más datos que siguen idénticamente la misma distribución no aporta "información nueva" para corregir el desbalance, aunque sí logró una estabilización marginal en el F1-Score de la clase minoritaria.

---

## Estructura del Repositorio

```text
bayesian-network-inference/
├── data/
│   ├── raw/                  # Dataset original limpio
│   ├── processed/            # Sets de Train/Test y Augmented
│   └── synthetic/            # Datos generados por la red
├── notebooks/
│   └── 01_EDA_and_Discretization.ipynb  # Análisis visual y justificación
├── src/                      # Scripts ejecutables
│   ├── data_prep.py
│   ├── structure_learning.py
│   ├── inference.py
│   ├── synthetic_gen.py
│   └── evaluate_augmented.py
├── results/
│   ├── figures/              # DAGs en formato PNG
│   └── models/               # Archivos .pkl con las redes entrenadas
├── requirements.txt
└── README.md
```

## Reproducibilidad (Instrucciones para Linux/Ubuntu)

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

1. **Clonar el repositorio:**

```bash
git clone [https://github.com/TU_USUARIO/bayesian-network-inference.git](https://github.com/TU_USUARIO/bayesian-network-inference.git)
cd bayesian-network-inference
```

2. **Crear y activar el entorno virtual:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

4. **Ejecutar el pipeline en orden:**

```bash
python src/data_prep.py
python src/structure_learning.py
python src/inference.py
python src/synthetic_gen.py
python src/evaluate_augmented.py
```
