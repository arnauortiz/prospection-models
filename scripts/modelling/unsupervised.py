import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.utils import plot_pr_curves
from visualization.statistics import tsne_visualization_fp


def save_metrics_to_file(metrics_dict, output_path):
    """Guarda les mètriques en un arxiu de text."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("📊 MÈTRIQUES DELS MODELS NO SUPERVISATS\n")
        f.write("=" * 50 + "\n\n")

        for model_name, metrics in metrics_dict.items():
            f.write(f"🧪 {model_name}\n")
            f.write("-" * 30 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name:<12}: {value:.4f}\n")
            f.write("\n")


def main_unsupervised(dataset):
    print("\n" + "=" * 50)
    print("🔍 ANÁLISIS NO SUPERVISADO")
    print("=" * 50)

    print("\n📊 Calculando métricas para modelos no supervisados...")

    # Umbrales óptimos (pueden ajustarse según necesidad)
    threshold_keywords = 0.3
    threshold_cosine = 0.5

    # Asegurarnos que target es booleano
    y_true = dataset["target"].astype(bool)

    # Para el modelo de palabras clave, necesitamos calcular el score promedio
    scores_keywords = dataset["score"].apply(
        lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x
    )
    y_pred_keywords = (scores_keywords >= threshold_keywords).astype(bool)

    # Para el modelo de similitud coseno
    similarities = dataset["target_similarity"].apply(
        lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x
    )
    y_pred_cosine = (similarities >= threshold_cosine).astype(bool)

    print("\nDistribución de predicciones:")
    print(
        f"Palabras clave - Positivos: {y_pred_keywords.sum()}, Negativos: {len(y_pred_keywords) - y_pred_keywords.sum()}"
    )
    print(
        f"Similitud - Positivos: {y_pred_cosine.sum()}, Negativos: {len(y_pred_cosine) - y_pred_cosine.sum()}"
    )

    # Métricas para modelo de palabras clave
    try:
        precision_kw = precision_score(y_true, y_pred_keywords, zero_division=0)
        recall_kw = recall_score(y_true, y_pred_keywords, zero_division=0)
        f1_kw = f1_score(y_true, y_pred_keywords, zero_division=0)
        acc_kw = accuracy_score(y_true, y_pred_keywords)
    except Exception as e:
        print(f"Error en métricas de palabras clave: {e}")
        precision_kw = recall_kw = f1_kw = acc_kw = 0

    # Métricas para modelo de similitud coseno
    try:
        precision_cos = precision_score(y_true, y_pred_cosine, zero_division=0)
        recall_cos = recall_score(y_true, y_pred_cosine, zero_division=0)
        f1_cos = f1_score(y_true, y_pred_cosine, zero_division=0)
        acc_cos = accuracy_score(y_true, y_pred_cosine)
    except Exception as e:
        print(f"Error en métricas de similitud: {e}")
        precision_cos = recall_cos = f1_cos = acc_cos = 0

    # Guardar métricas en diccionario
    metrics_dict = {
        "Model de Paraules Clau": {
            "Precision": precision_kw,
            "Recall": recall_kw,
            "F1 Score": f1_kw,
            "Accuracy": acc_kw,
        },
        "Model de Similitud": {
            "Precision": precision_cos,
            "Recall": recall_cos,
            "F1 Score": f1_cos,
            "Accuracy": acc_cos,
        },
    }

    # Guardar métricas en archivo
    save_metrics_to_file(metrics_dict, "output/unsupervised_metrics.txt")
    print("✅ Mètriques guardades a output/unsupervised_metrics.txt")

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
    pred_sim = similarities >= threshold
    # Crear una copia explícita del DataFrame
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

    # Finetuning de thresholds
    print("\n" + "=" * 50)
    print("🔍 FINETUNING DE THRESHOLDS SIMILITUD")
    print("=" * 50)

    # Probar diferentes thresholds con steps más pequeños
    thresholds = np.arange(0.5, 0.61, 0.01)  # De 0.5 a 0.6 con steps de 0.01
    results = []

    for threshold in thresholds:
        y_pred = (similarities >= threshold).astype(bool)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Accuracy": acc,
            }
        )

    # Convertir a DataFrame para mejor visualización
    results_df = pd.DataFrame(results)

    # Encontrar el mejor threshold basado en F1 Score
    best_idx = results_df["F1 Score"].idxmax()
    best_threshold = results_df.loc[best_idx, "Threshold"]
    best_f1 = results_df.loc[best_idx, "F1 Score"]

    print("\n📊 Resultados del Finetuning:")
    print(results_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    print(f"\n✨ Millor threshold: {best_threshold:.2f}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"   Precision: {results_df.loc[best_idx, 'Precision']:.4f}")
    print(f"   Recall: {results_df.loc[best_idx, 'Recall']:.4f}")
    print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")

    # Añadir resultados del finetuning al archivo de métricas
    with open("output/unsupervised_metrics.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("🔍 FINETUNING DE THRESHOLDS\n")
        f.write("=" * 50 + "\n\n")
        f.write(
            results_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x))
        )
        f.write(f"\n\n✨ Millor threshold: {best_threshold:.2f}\n")
        f.write(f"   F1 Score: {best_f1:.4f}\n")
        f.write(f"   Precision: {results_df.loc[best_idx, 'Precision']:.4f}\n")
        f.write(f"   Recall: {results_df.loc[best_idx, 'Recall']:.4f}\n")
        f.write(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}\n")
