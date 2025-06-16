# Prospection Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene modelos de clasificación para la prospección de empresas, incluyendo tanto modelos supervisados como no supervisados.

## 📋 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/yourusername/prospection-models.git
cd prospection-models
```

2. Crear y activar el entorno virtual:
```bash
# En Windows
python -m venv venv
.\venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
prospection-models/
│
├── data/
│   ├── raw/          # Datos originales
│   └── interim/      # Datos procesados
│
├── output/           # Resultados y visualizaciones
│   ├── histograms_before.png
│   ├── histograms_after.png
│   ├── confusion_matrices.png
│   ├── unsupervised_pr_curves.png
│   └── tsne_visualization_fp.png
│
├── scripts/
│   ├── modelling/    # Modelos supervisados y no supervisados
│   │   ├── supervised.py
│   │   └── unsupervised.py
│   │
│   ├── preprocessing/# Procesamiento de datos
│   │   └── data_loading.py
│   │
│   └── visualization/# Visualizaciones y análisis
│       ├── utils.py
│       └── statistics.py
│
├── requirements.txt  # Dependencias del proyecto
└── README.md        # Este archivo
```

## ⚠️ Nota Importante sobre los Datos

Los datos originales no están incluidos en este repositorio por razones de privacidad y confidencialidad. Sin embargo, puedes ver los resultados del análisis y las visualizaciones en la carpeta `output/`, que incluye:

- Curvas Precision-Recall
- Matrices de confusión
- Visualizaciones t-SNE
- Histogramas
- Métricas de rendimiento

## 🎯 Uso

Para ejecutar el análisis completo:

```bash
python scripts/main.py
```

Para ejecutar componentes específicos:

```bash
# Modelos supervisados
python scripts/modelling/supervised.py

# Modelos no supervisados
python scripts/modelling/unsupervised.py

# Análisis estadístico
python scripts/visualization/statistics.py
```

## 📊 Resultados

Los resultados se guardan automáticamente en la carpeta `output/`:

| Archivo | Descripción |
|---------|-------------|
| `histograms_before.png` | Histogramas antes del preprocesamiento |
| `histograms_after.png` | Histogramas después del preprocesamiento |
| `confusion_matrices.png` | Matrices de confusión de los modelos supervisados |
| `unsupervised_pr_curves.png` | Curvas Precision-Recall de los modelos no supervisados |
| `tsne_visualization_fp.png` | Visualización t-SNE con falsos positivos |

## 🔧 Dependencias Principales

| Paquete | Uso |
|---------|-----|
| pandas | Manipulación de datos |
| numpy | Cálculos numéricos |
| scikit-learn | Modelos de machine learning |
| xgboost | Modelo de boosting |
| matplotlib | Visualizaciones |
| seaborn | Visualizaciones estadísticas |

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.