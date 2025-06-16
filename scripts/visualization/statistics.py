from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import seaborn as sns

def statistics(dataset,output_path):
    # Convertir tipus per seguretat
    dataset['score'] = dataset['score'].astype(float)
    dataset['target_similarity'] = dataset['target_similarity'].astype(float)
    dataset['target'] = dataset['target'].astype(bool)

    # Ordenem les dades
    df_score = dataset.sort_values(by='score').reset_index(drop=True)
    df_sim = dataset.sort_values(by='target_similarity').reset_index(drop=True)

    # Estadístiques
    med_score = df_score['score'].median()
    mean_score = df_score['score'].mean()
    std_score = df_score['score'].std()

    med_sim = df_sim['target_similarity'].median()
    mean_sim = df_sim['target_similarity'].mean()
    std_sim = df_sim['target_similarity'].std()

    # 🎨 Crear gràfics un sota l'altre
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Llegenda comuna
    legend_elements = [
        mpatches.Patch(color='red', label='Target'),
        mpatches.Patch(color='gray', label='No Target'),
        mpatches.Patch(facecolor='blue', edgecolor='blue', alpha=0.1, label='±1σ'),
        mpatches.Patch(facecolor='none', edgecolor='blue', linestyle='--', label='Mediana')
    ]

    # ▪️ SCORE
    colors_score = ['red' if t else 'gray' for t in df_score['target']]
    axs[0].scatter(df_score.index, df_score['score'], c=colors_score, edgecolor='black', alpha=0.7, s=40)
    axs[0].axhline(med_score, color='blue', linestyle='--')
    axs[0].fill_between(df_score.index, med_score - std_score, med_score + std_score, color='blue', alpha=0.1)
    axs[0].set_title(f'Score per paraula clau | Mediana: {med_score:.3f} | Mitjana: {mean_score:.3f} | ±1σ: {std_score:.3f}')
    axs[0].set_ylabel('Score')
    axs[0].legend(handles=legend_elements, loc='upper left')

    # ▪️ SIMILARITY
    colors_sim = ['red' if t else 'gray' for t in df_sim['target']]
    axs[1].scatter(df_sim.index, df_sim['target_similarity'], c=colors_sim, edgecolor='black', alpha=0.7, s=40)
    axs[1].axhline(med_sim, color='blue', linestyle='--')
    axs[1].fill_between(df_sim.index, med_sim - std_sim, med_sim + std_sim, color='blue', alpha=0.1)
    axs[1].set_title(f'Similitud coseno | Mediana: {med_sim:.3f} | Mitjana: {mean_sim:.3f} | ±1σ: {std_sim:.3f}')
    axs[1].set_xlabel('Índex ordenat')
    axs[1].set_ylabel('Similitud')
    axs[1].legend(handles=legend_elements, loc='upper left')

    #Guardar gràfic
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
def tsne_visualization(dataset, output_path):
   
    # Si ja tens arrays reals de numpy:
    X = np.vstack(dataset['description_embedding'].values)

    # Reducció a 2D amb t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    X_2d = tsne.fit_transform(X)

    # Afegim les coordenades al dataset
    dataset['tsne_1'] = X_2d[:, 0]
    dataset['tsne_2'] = X_2d[:, 1]

    # Visualització
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=dataset,
        x='tsne_1', y='tsne_2',
        hue='sector',
        palette='tab20',
        alpha=0.85
    )
    plt.title('Visualització 2D dels embeddings per sector')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Guardar gràfic
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def tsne_visualization_fp(combined_df, output_path):

    # 🔹 Convertir embeddings
    X = np.vstack(combined_df['description_embedding'].values)

    # 🔹 Aplicar t-SNE a tot el conjunt
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    X_2d = tsne.fit_transform(X)

    # 🔹 Afegir coordenades
    combined_df['tsne_1'] = X_2d[:, 0]
    combined_df['tsne_2'] = X_2d[:, 1]

    # 🔹 Separar els dos conjunts
    background_df = combined_df[~combined_df['is_fp']]
    falsos_df = combined_df[combined_df['is_fp']]

    # 🔹 Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=background_df,
        x='tsne_1', y='tsne_2',
        hue='sector',
        palette='tab20',
        alpha=0.85
    )

    # 🔺 Afegir falsos positius amb marcador vermell buit
    plt.scatter(
        falsos_df['tsne_1'],
        falsos_df['tsne_2'],
        edgecolor='red',
        facecolor='none',
        linewidth=2,
        s=150,
        label='Falsos Positius'
    )

    plt.title('Visualització 2D dels embeddings per sector (amb falsos positius marcats)')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
