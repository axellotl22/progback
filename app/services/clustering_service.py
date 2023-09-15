"""
Dienste für Clustering-Funktionen.
"""

import logging
import os
from typing import List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Lädt eine Datei in ein Pandas DataFrame."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file type")

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt das DataFrame von leeren und unvollständigen Zeilen."""
    return data_frame.dropna()

def kmeans_clustering(data_frame: pd.DataFrame, 
                      n_clusters: int, 
                      random_state: int) -> Tuple[float, List[int]]:
    
    """Führt KMeans-Clustering aus und gibt die Trägheit und Labels zurück."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, 
                    n_init=10, random_state=random_state)
    kmeans.fit(data_frame)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(data_frame, kmeans.labels_)
    return inertia, sil_score

def determine_optimal_clusters(data_frame: pd.DataFrame) -> int:
    """Bestimmt die optimale Clusteranzahl mittels Elbogen-Methode und Silhouettenmethode."""
    wcss = []
    sil_scores = []
    max_clusters = min(data_frame.shape[0] - 1, MAX_CLUSTERS)

    for i in range(2, max_clusters):
        inertia, sil_score = kmeans_clustering(data_frame, i, 0)
        wcss.append(inertia)
        sil_scores.append(sil_score)

    # Silhouettenmethode
    silhouette_point = list(range(2, max_clusters))[sil_scores.index(max(sil_scores))]

    # Hier könnte eine verbesserte Logik für die Elbogen-Methode hinzugefügt werden
    # Für den Moment nehmen wir einfach den Silhouettenpunkt
    return silhouette_point

def perform_clustering(data_frame: pd.DataFrame, n_clusters: int) -> List[int]:
    """Führt KMeans-Clustering auf dem DataFrame aus."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(data_frame)
    return kmeans.labels_.tolist()

def delete_file(file_path: str):
    """Löscht die angegebene Datei."""
    try:
        os.remove(file_path)
        logging.info("File %s successfully deleted.", file_path)
    except OSError as error:
        logging.error("Error deleting file %s: %s", file_path, error)
