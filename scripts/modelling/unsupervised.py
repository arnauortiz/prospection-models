import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.utils import plot_pr_curves
from visualization.statistics import tsne_visualization_fp


def main_unsupervised(dataset):
    print("\n" + "=" * 50)
    print("🔍 ANÁLISIS NO SUPERVISADO")
    print("=" * 50)

    print("\n📊 Calculando métricas para modelos no supervisados...")

    # Umbrales óptimos (pueden ajustarse según necesidad)
    threshold_keywords = 0.3
    threshold_cosine = 0.5

    # Predicciones
    y_true = dataset["target"].astype(bool)
    y_pred_keywords = dataset["score"] >= threshold_keywords
    y_pred_cosine = dataset["target_similarity"] >= threshold_cosine

    # Métricas para modelo de palabras clave
    precision_kw = precision_score(y_true, y_pred_keywords)
    recall_kw = recall_score(y_true, y_pred_keywords)
    f1_kw = f1_score(y_true, y_pred_keywords)
    acc_kw = accuracy_score(y_true, y_pred_keywords)

    # Métricas para modelo de similitud coseno
    precision_cos = precision_score(y_true, y_pred_cosine)
    recall_cos = recall_score(y_true, y_pred_cosine)
    f1_cos = f1_score(y_true, y_pred_cosine)
    acc_cos = accuracy_score(y_true, y_pred_cosine)

    print("\n" + "-" * 50)
    print("📈 MÉTRICAS DE MODELOS NO SUPERVISADOS")
    print("-" * 50)

    print("\n🔍 Modelo de Palabras Clave (threshold = 0.3):")
    print(f"   Precision : {precision_kw:.4f}")
    print(f"   Recall    : {recall_kw:.4f}")
    print(f"   F1 Score  : {f1_kw:.4f}")
    print(f"   Accuracy  : {acc_kw:.4f}")

    print("\n🔍 Modelo de Similitud Coseno (threshold = 0.5):")
    print(f"   Precision : {precision_cos:.4f}")
    print(f"   Recall    : {recall_cos:.4f}")
    print(f"   F1 Score  : {f1_cos:.4f}")
    print(f"   Accuracy  : {acc_cos:.4f}")

    print("\n📊 Generando curvas Precision-Recall...")
    plot_pr_curves(dataset, "output/unsupervised_pr_curves.png")

    print("\n🎯 Analizando falsos positivos...")
    threshold = 0.5
    pred_sim = dataset["target_similarity"] >= threshold
    false_positives_sim = dataset[
        (pred_sim == True) & (dataset["target"] == False)
    ].copy()
    dataset["is_fp"] = False
    false_positives_sim.loc[:, "is_fp"] = True

    print(f"✅ Falsos positivos encontrados: {len(false_positives_sim)}")

    print("\n📈 Generando visualización t-SNE...")
    # 🔹 Concatenar
    combined_df = pd.concat([dataset, false_positives_sim], ignore_index=True)
    tsne_visualization_fp(combined_df, "output/tsne_visualization_fp.png")
    print("✅ Visualización t-SNE guardada en output/tsne_visualization_fp.png")
