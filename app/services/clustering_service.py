import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Lädt eine Datei in ein Pandas DataFrame."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt das DataFrame von leeren und unvollständigen Zeilen."""
    df.dropna(inplace=True)
    # Weitere Bereinigungslogik kann hier hinzugefügt werden
    return df

def determine_optimal_clusters(df: pd.DataFrame) -> int:
    """Bestimmt die optimale Clusteranzahl mittels Elbogen-Methode und Silhouettenmethode."""
    wcss = []
    sil_scores = []
    max_clusters = min(df.shape[0] - 1, 10)  # Begrenzen der Anzahl der Cluster auf 10 oder weniger, je nach Größe des DataFrames
    
    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(df, kmeans.labels_))

    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(df, kmeans.labels_))

    # Elbogen-Methode
    elbow_point = [i for i in range(2, max_clusters)][wcss.index(min(wcss))]

    # Silhouettenmethode
    silhouette_point = [i for i in range(2, max_clusters)][sil_scores.index(max(sil_scores))]

    # Mittelwert beider Methoden
    optimal_clusters = (elbow_point + silhouette_point) // 2

    return optimal_clusters

def perform_clustering(df: pd.DataFrame, n_clusters: int) -> List[int]:
    """Führt KMeans-Clustering auf dem DataFrame aus."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(df)
    return kmeans.labels_.tolist()

def delete_file(file_path: str):
    """Löscht die angegebene Datei."""
    try:
        os.remove(file_path)
        logging.info(f"File {file_path} successfully deleted.")
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {e}")