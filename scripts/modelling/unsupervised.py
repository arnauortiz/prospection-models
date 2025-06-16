import pandas as pd
from sklearn.model_selection import train_test_split

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.utils import plot_pr_curves
from visualization.statistics import tsne_visualization_fp

def main_unsupervised(dataset):
    plot_pr_curves(dataset, "output/unsupervised_pr_curves.png")
    threshold = 0.5
    pred_sim = dataset['target_similarity'] >= threshold
    false_positives_sim = dataset[(pred_sim == True) & (dataset['target'] == False)]
    dataset['is_fp'] = False
    false_positives_sim['is_fp'] = True

    # 🔹 Concatenar
    combined_df = pd.concat([dataset, false_positives_sim], ignore_index=True)
    tsne_visualization_fp(combined_df, "output/tsne_visualization_fp.png")