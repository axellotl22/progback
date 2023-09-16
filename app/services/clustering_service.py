"""
Dienste für Clustering-Funktionen.
"""

import os
import logging
import numpy as np
import pandas as pd
from gap_statistic import OptimalK
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10

def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Lädt eine Datei in ein Pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    if file_path.endswith('.json'):
        return pd.read_json(file_path)
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}")

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame von leeren und unvollständigen Zeilen.
    """
    return data_frame.dropna()

def determine_optimal_clusters(data_frame: pd.DataFrame) -> int:
    """
    Bestimmt die optimale Clusteranzahl mittels verschiedener Methoden.
    """
    if len(data_frame) < 1000:
        optimal_k = OptimalK(parallel_backend='joblib')
        n_clusters = optimal_k(
            data_frame.values, cluster_array=np.arange(1, min(MAX_CLUSTERS, len(data_frame))))
        return n_clusters

    n_clusters = determine_clusters_using_silhouette(data_frame)
    return min(n_clusters, len(data_frame) - 1)

def determine_clusters_using_silhouette(data_frame: pd.DataFrame) -> int:
    """
    Bestimmt die optimale Clusteranzahl mittels der Silhouetten-Methode.
    """
    sil_scores = []
    max_clusters = min(data_frame.shape[0] - 1, MAX_CLUSTERS)

    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0).fit(data_frame)
        sil_scores.append(silhouette_score(data_frame, kmeans.labels_))

    return list(range(2, max_clusters))[sil_scores.index(max(sil_scores))]

def select_columns(data_frame: pd.DataFrame, columns: List[int]) -> pd.DataFrame:
    """
    Wählt bestimmte Spalten aus einem DataFrame aus basierend auf deren Index.
    """
    if any(col_idx >= len(data_frame.columns) for col_idx in columns):
        raise ValueError(f"Ungültiger Spaltenindex. Das DataFrame hat nur {len(data_frame.columns)} Spalten.")
    
    selected_columns = [data_frame.columns[idx] for idx in columns]
    return data_frame[selected_columns]


def perform_clustering(data_frame: pd.DataFrame, n_clusters: int) -> dict:
    """
    Führt KMeans-Clustering auf dem DataFrame aus und gibt die Ergebnisse im gewünschten Format zurück.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=MAX_CLUSTERS)
    kmeans.fit(data_frame)

    points = [{"x": point[0], "y": point[1], "cluster": label}
              for point, label in zip(data_frame.values, kmeans.labels_)]
    centroids = [{"x": centroid[0], "y": centroid[1], "cluster": idx}
                 for idx, centroid in enumerate(kmeans.cluster_centers_)]

    response_data = {
        "points": points,
        "centroids": centroids,
        "point_to_centroid_mappings": dict(enumerate(kmeans.labels_)),
        "x_label": data_frame.columns[0],
        "y_label": data_frame.columns[1]
    }
    return response_data

def delete_file(file_path: str):
    """
    Löscht die angegebene Datei.
    """
    try:
        if os.environ.get("TEST_MODE") != "True":
            os.remove(file_path)
            logging.info("File %s successfully deleted.", file_path)
    except FileNotFoundError:
        logging.warning("File %s was already deleted.", file_path)
    except OSError as error:
        logging.error("Error deleting file %s: %s", file_path, error)
