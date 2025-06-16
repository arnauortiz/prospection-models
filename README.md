# Models de Prospecció

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Aquest repositori conté models de classificació per a la prospecció d'empreses, incloent-hi tant models supervisats com no supervisats.

## 📋 Requisits

- Python 3.8 o superior
- pip (gestor de paquets de Python)

## 🚀 Instal·lació

1. Clonar el repositori:
```bash
git clone https://github.com/yourusername/prospection-models.git
cd prospection-models
```

2. Crear i activar l'entorn virtual:
```bash
# A Windows
python -m venv venv
.\venv\Scripts\activate

# A Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Instal·lar les dependències:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Projecte

```
prospection-models/
│
├── data/
│   ├── raw/          # Dades originals
│   └── interim/      # Dades processades
│
├── output/           # Resultats i visualitzacions
│   ├── histograms_before.png
│   ├── histograms_after.png
│   ├── confusion_matrices.png
│   ├── unsupervised_pr_curves.png
│   └── tsne_visualization_fp.png
│
├── scripts/
│   ├── modelling/    # Models supervisats i no supervisats
│   │   ├── supervised.py
│   │   └── unsupervised.py
│   │
│   ├── preprocessing/# Processament de dades
│   │   └── data_loading.py
│   │
│   └── visualization/# Visualitzacions i anàlisi
│       ├── utils.py
│       └── statistics.py
│
├── requirements.txt  # Dependències del projecte
└── README.md        # Aquest arxiu
```

## ⚠️ Nota Important sobre les Dades

Les dades originals no estan incloses en aquest repositori per raons de privacitat i confidencialitat. Tot i així, pots veure els resultats de l'anàlisi i les visualitzacions a la carpeta `output/`, que inclou:

- Corbes Precision-Recall
- Matrius de confusió
- Visualitzacions t-SNE
- Histogrames
- Mètriques de rendiment

## 🎯 Ús

Per executar l'anàlisi complet:

```bash
python scripts/main.py
```

Per executar components específics:

```bash
# Models supervisats
python scripts/modelling/supervised.py

# Models no supervisats
python scripts/modelling/unsupervised.py

# Anàlisi estadístic
python scripts/visualization/statistics.py
```

## 📊 Resultats

Els resultats es guarden automàticament a la carpeta `output/`:

| Arxiu | Descripció |
|---------|-------------|
| `histograms_before.png` | Histogrames abans del preprocessament |
| `histograms_after.png` | Histogrames després del preprocessament |
| `confusion_matrices.png` | Matrius de confusió dels models supervisats |
| `unsupervised_pr_curves.png` | Corbes Precision-Recall dels models no supervisats |
| `tsne_visualization_fp.png` | Visualització t-SNE amb falsos positius |

## 🔧 Dependències Principals

| Paquet | Ús |
|---------|-----|
| pandas | Manipulació de dades |
| numpy | Càlculs numèrics |
| scikit-learn | Models de machine learning |
| xgboost | Model de boosting |
| matplotlib | Visualitzacions |
| seaborn | Visualitzacions estadístiques |

## 💻 Ús
1. Carrega les dades:
```bash
python scripts/data_loading.py
```

2. Executa els models supervisats:
```bash
python scripts/modelling/supervised.py
```

3. Executa els models no supervisats:
```bash
python scripts/modelling/unsupervised.py
```

4. Genera visualitzacions:
```bash
python scripts/visualization/statistics.py
```

## 📊 Resultats

### Models Supervisats
| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|---------|-----------|-----------|
| RandomForest | 1.0000 | 0.9630 | 0.9811 | 0.9828 |
| LogisticRegression | 1.0000 | 0.9630 | 0.9811 | 0.9828 |
| SVC (linear) | 1.0000 | 0.9630 | 0.9811 | 0.9828 |
| KNN | 0.9643 | 1.0000 | 0.9818 | 0.9828 |
| NaiveBayes | 1.0000 | 0.9630 | 0.9811 | 0.9828 |
| XGBoost | 1.0000 | 0.9259 | 0.9615 | 0.9655 |

### Models No Supervisats
| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|---------|-----------|-----------|
| Model de Paraules Clau | 0.9273 | 0.7612 | 0.8364 | 0.8601 |
| Model de Similitud | 0.9710 | 1.0000 | 0.9853 | 0.9861 |

### Finetuning del Model de Similitud
El millor threshold trobat és **0.55** amb les següents mètriques:
- F1 Score: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- Accuracy: 1.0000

## 📦 Dependències Principals
| Paquet | Versió | Ús |
|--------|---------|-----|
| pandas | 2.0.0 | Manipulació de dades |
| numpy | 1.24.0 | Càlculs numèrics |
| scikit-learn | 1.2.0 | Models de machine learning |
| xgboost | 1.7.0 | Model XGBoost |
| matplotlib | 3.7.0 | Visualitzacions |
| seaborn | 0.12.0 | Visualitzacions estadístiques |