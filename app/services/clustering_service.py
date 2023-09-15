"""
Dienste für Clustering-Funktionen.
"""

import os
import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Lädt eine Datei in ein Pandas DataFrame.

    Args:
    - file_path (str): Pfad zur Datei.

    Returns:
    - pd.DataFrame: Geladenes DataFrame.
    """

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)

    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)

    raise ValueError("Unsupported file type")


def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame von leeren und unvollständigen Zeilen.

    Args:
    - data_frame (pd.DataFrame): Das zu bereinigende DataFrame.

    Returns:
    - pd.DataFrame: Bereinigtes DataFrame.
    """
    return data_frame.dropna()


def determine_optimal_clusters(data_frame: pd.DataFrame) -> int:
    """
    Bestimmt die optimale Clusteranzahl mittels Elbogen-Methode und Silhouettenmethode.

    Args:
    - data_frame (pd.DataFrame): DataFrame mit Datenpunkten.

    Returns:
    - int: Die optimale Anzahl von Clustern.
    """

    def kmeans_clustering(data: pd.DataFrame, n_clusters: int) -> KMeans:
        """Führt KMeans-Clustering aus und gibt das KMeans-Objekt zurück."""
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        return kmeans

    sil_scores = []
    max_clusters = min(data_frame.shape[0] - 1, MAX_CLUSTERS)

    for i in range(2, max_clusters):
        kmeans = kmeans_clustering(data_frame, i)
        sil_scores.append(silhouette_score(data_frame, kmeans.labels_))

    return list(range(2, max_clusters))[sil_scores.index(max(sil_scores))]


def perform_clustering(data_frame: pd.DataFrame, n_clusters: int) -> KMeans:
    """
    Führt KMeans-Clustering auf dem DataFrame aus.

    Args:
    - data_frame (pd.DataFrame): DataFrame mit Datenpunkten.
    - n_clusters (int): Anzahl der gewünschten Cluster.

    Returns:
    - KMeans: Trainiertes KMeans-Objekt.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(data_frame)
    return kmeans


def delete_file(file_path: str):
    """
    Löscht die angegebene Datei.

    Args:
    - file_path (str): Pfad zur zu löschenden Datei.
    """
    try:
        if os.environ.get("TEST_MODE") != "True":
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info("File %s successfully deleted.", file_path)
    except OSError as error:
        logging.error("Error deleting file %s: %s", file_path, error)
