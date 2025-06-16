import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization.utils import plot_confusion_matrices


def split_data(dataset, target_col="target", test_size=0.4, random_state=42):
    X = dataset["description_embedding"].tolist()
    y = dataset["target"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    X = np.array(X_train)
    y = np.array(y_train)

    # Modelos a comparar
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "SVC (linear)": SVC(kernel="linear", class_weight="balanced", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=1),
    }

    # Resultados acumulados
    results = {name: {"f1": [], "accuracy": []} for name in models.keys()}

    print("\n" + "=" * 50)
    print("🤖 ENTRENAMIENTO DE MODELOS")
    print("=" * 50)

    # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n🔍 Evaluando modelo: {name}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred, pos_label=True, average="binary")
            acc = accuracy_score(y_test, y_pred)

            results[name]["f1"].append(f1)
            results[name]["accuracy"].append(acc)

            print(f"  Fold {fold}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")

    print("\n" + "-" * 50)
    print("📊 RESUMEN DE RESULTADOS")
    print("-" * 50)

    for name, metrics in results.items():
        f1_avg = np.mean(metrics["f1"])
        acc_avg = np.mean(metrics["accuracy"])
        print(f"✅ {name:<20} F1: {f1_avg:.4f} | Accuracy: {acc_avg:.4f}")

    return models


def evaluate_model(models, X_test, y_test):
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Diccionari per guardar resultats
    final_results = {}

    print("\n" + "=" * 50)
    print("📈 EVALUACIÓN FINAL")
    print("=" * 50)
    print("\n🔍 Resultados sobre el conjunto de test (holdout):\n")

    for name, model in models.items():
        print(f"🧪 {name}")

        # Predicció sobre test
        y_pred = model.predict(X_test)

        # Mètriques
        precision = precision_score(y_test, y_pred, pos_label=True)
        recall = recall_score(y_test, y_pred, pos_label=True)
        f1 = f1_score(y_test, y_pred, pos_label=True)
        acc = accuracy_score(y_test, y_pred)

        final_results[name] = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": acc,
        }

        print(f"   Precision : {precision:.4f}")
        print(f"   Recall    : {recall:.4f}")
        print(f"   F1 Score  : {f1:.4f}")
        print(f"   Accuracy  : {acc:.4f}")
        print()

    # 📋 També pots imprimir un resum estil taula
    import pandas as pd

    df_results = pd.DataFrame(final_results).T
    print("\n" + "-" * 50)
    print("📊 RESUMEN FINAL")
    print("-" * 50)
    print(df_results.round(4))


def main_supervised(dataset):
    X_train, X_test, y_train, y_test = split_data(dataset)
    models = train_models(X_train, y_train)
    evaluate_model(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test, "output/confusion_matrices.png")
