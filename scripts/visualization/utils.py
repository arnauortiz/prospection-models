import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

def plot_hsitograms(dataset, output_path):
    # Convertim a float per assegurar-nos
    scores = np.array(dataset['score'].values, dtype=float)
    similarities = np.array(dataset['target_similarity'].values, dtype=float)

    # Crear dos histogrames en paral·lel
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Histograma de score
    axs[0].hist(scores, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title('Distribució de score per paraula clau')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Freqüència')

    # Histograma de target_similarity
    axs[1].hist(similarities, bins=20, color='lightgreen', edgecolor='black')
    axs[1].set_title('Distribució de la similitud coseno')
    axs[1].set_xlabel('Similitud')
    
    # Guardar la figura
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
def plot_pr_curves(dataset, output_path):
    # Inputs
    y_true = dataset['target'].astype(bool).values
    scores_keywords = dataset['score'].astype(float).values
    scores_cosine = dataset['target_similarity'].astype(float).values

    # PR curves
    prec_k, rec_k, _ = precision_recall_curve(y_true, scores_keywords)
    auc_k = auc(rec_k, prec_k)

    prec_c, rec_c, _ = precision_recall_curve(y_true, scores_cosine)
    auc_c = auc(rec_c, prec_c)

    # Prediccions binàries
    y_pred_keywords = scores_keywords >= 0.3
    y_pred_cosine = scores_cosine >= 0.5

    # Layout (PR + dues matrius verticals)
    fig = plt.figure(figsize=(14, 8))

    # PR curve a l'esquerra (ocupa 2 files x 2 columnes)
    ax_pr = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax_pr.plot(rec_k, prec_k, label=f'Paraules clau (AUC = {auc_k:.3f})', color='navy')
    ax_pr.plot(rec_c, prec_c, label=f'Similitud coseno (AUC = {auc_c:.3f})', color='darkgreen')
    ax_pr.set_title('Curves Precision-Recall: Mètodes no supervisats')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True)

    # Matriu 1 a dalt a la dreta
    ax_kw = plt.subplot2grid((2, 3), (0, 2))
    cm_kw = confusion_matrix(y_true, y_pred_keywords)
    sns.heatmap(cm_kw, annot=True, fmt='d', cmap='Blues', cbar=False,
                ax=ax_kw, xticklabels=['No Target', 'Target'], yticklabels=['No Target', 'Target'])
    ax_kw.set_title('Confusió - Paraules clau')
    ax_kw.set_xlabel('Predicted')
    ax_kw.set_ylabel('Actual')

    # Matriu 2 a sota a la dreta
    ax_cos = plt.subplot2grid((2, 3), (1, 2))
    cm_cos = confusion_matrix(y_true, y_pred_cosine)
    sns.heatmap(cm_cos, annot=True, fmt='d', cmap='Greens', cbar=False,
                ax=ax_cos, xticklabels=['No Target', 'Target'], yticklabels=['No Target', 'Target'])
    ax_cos.set_title('Confusió - Similitud coseno')
    ax_cos.set_xlabel('Predicted')
    ax_cos.set_ylabel('Actual')
    
    # Guardar gràfic
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_confusion_matrices(models, X_test, y_test, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    ax=axes[idx], xticklabels=['No Target', 'Target'],
                    yticklabels=['No Target', 'Target'])
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    # Si sobren subplots buits
    for i in range(len(models), 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)