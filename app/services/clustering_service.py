"""
Dienste für Clustering-Funktionen.
"""
from typing import List
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import os
import logging


logging.basicConfig(level=logging.INFO)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Lädt eine Datei in ein Pandas DataFrame."""
    if file_path.endswith('.csv'):
        data_frame = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data_frame = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")
    return data_frame

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt das DataFrame von leeren und unvollständigen Zeilen."""
    data_frame.dropna(inplace=True)
    # Weitere Bereinigungslogik kann hier hinzugefügt werden
    return data_frame

def determine_optimal_clusters(data_frame: pd.DataFrame) -> int:
    """Bestimmt die optimale Clusteranzahl mittels Elbogen-Methode und Silhouettenmethode."""
    wcss = []
    sil_scores = []
    # Begrenzen der Anzahl der Cluster auf 10 oder weniger, je nach Größe des DataFrames
    max_clusters = min(data_frame.shape[0] - 1, 10)  
    
    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_frame)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(data_frame, kmeans.labels_))

    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data_frame)
        wcss.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(data_frame, kmeans.labels_))

    # Elbogen-Methode
    elbow_point = list(range(2, max_clusters))

    # Silhouettenmethode
    silhouette_point = [i for i in range(2, max_clusters)][sil_scores.index(max(sil_scores))]

    # Mittelwert beider Methoden
    optimal_clusters = (elbow_point + silhouette_point) // 2

    return optimal_clusters

def perform_clustering(data_frame: pd.DataFrame, n_clusters: int) -> List[int]:
    """Führt KMeans-Clustering auf dem DataFrame aus."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(data_frame)
    return kmeans.labels_.tolist()

def delete_file(file_path: str):
    """Löscht die angegebene Datei."""
    try:
        os.remove(file_path)
        logging.info(f"File {file_path} successfully deleted.")
    except Exception as error:
        logging.error(f"Error deleting file {file_path}: {error}")
