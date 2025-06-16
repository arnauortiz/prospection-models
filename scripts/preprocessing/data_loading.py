import pandas as pd
import ast 
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization.utils import plot_hsitograms


def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_parquet(file_path)
    return data

def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    dataset = df[["name","score","description","target_similarity","description_embedding","target"]].copy()
    dataset['description_embedding'] = dataset['description_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    dataset['target'] = dataset['target'].apply(lambda x: x == 'False')
    
    plot_hsitograms(dataset, "output/histograms_before.png")
    
    # Eliminar valor mas grande de score
    dataset['target_similarity'] = dataset['target_similarity'].apply(lambda x: np.array(ast.literal_eval(x)))
    dataset['score'] = dataset['score'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    
    score_max = dataset['score'].max()
    print("\n" + "="*50)
    print("🔍 Preprocesamiento de datos")
    print("="*50)

    score_max = dataset['score'].max()
    print(f"\n🔸 Max score antes del filtrado: {score_max:.4f}")
    dataset = dataset[dataset['score'] < score_max]
    score_max = dataset['score'].max()
    print(f"✅ Nuevo max score (después del filtrado): {score_max:.4f}")

    
    plot_hsitograms(dataset, "output/histograms_after.png")
    
    # Afegim sector al dataset
    dataset.loc[dataset['target'] == True, 'sector'] = "Telecomunicaciones"
    sectores_no_target = pd.read_csv('data/interim/empreses_sectors_1-200.csv')
    sectores_no_target.rename(columns={'Empresa': 'name'}, inplace=True)
    dataset = dataset.merge(
    sectores_no_target,
    on="name",
    how="left",
    suffixes=('', '_new')
    )

    # Si vols mantenir només el sector original per target=True
    dataset['sector'] = dataset.apply(
        lambda row: row['sector'] if row['target'] else row['Sector'],
        axis=1
    )

    # Elimina la columna auxiliar
    dataset.drop(columns=['Sector'], inplace=True)
    dataset = dataset.dropna()
    return dataset

def explore_data(dataset: pd.DataFrame):
    print("\n" + "="*50)
    print("📊 EXPLORACIÓN DEL DATASET")
    print("="*50)
    
    print("\n🔹 Primeras 5 filas del dataset:\n")
    print(dataset.head(5))
    
    print("\n" + "-"*50)
    print("🎯 Distribución de la variable objetivo (`target`):\n")
    print(dataset['target'].value_counts())
    
    print("\n" + "-"*50)
    print("🏷️ Distribución de sectores:\n")
    print(dataset['sector'].value_counts())
    
    print("\n" + "="*50 + "\n")
    
def main_dataset():
    data_path = "data/raw/companies.parquet"
    dataset = preprocess_data(data_path)
    explore_data(dataset)
    return dataset
    
if __name__ == "__main__":
    dataset = main_dataset()