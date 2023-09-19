"""
Dienste für Clustering-Funktionen.
"""

import logging
import os
from typing import List

import numpy as np
import pandas as pd
from gap_statistic import OptimalK
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10

# Distanz-Funktionen
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))


def minkowski_distance(x, y, p=2):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Custom KMeans mit Unterstützung für verschiedene Distanzmaße
class CustomKMeans:
    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.iterations_ = 0 

        if distance_metric == "EUCLIDEAN":
            self.distance = euclidean_distance
        elif distance_metric == "MANHATTAN":
            self.distance = manhattan_distance
        elif distance_metric == "CHEBYSHEV":
            self.distance = chebyshev_distance
        elif distance_metric == "MINKOWSKI":
            self.distance = minkowski_distance
        else:
            raise ValueError(f"Unbekanntes Distanzmaß: {distance_metric}")

    def fit(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        
        for iteration in range(self.max_iter):
            dists = np.array([[self.distance(x, center) for center in self.cluster_centers_] for x in X])
            labels = np.argmin(dists, axis=1)
            
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                self.labels_ = labels
                break
            
            self.cluster_centers_ = new_centers
            self.labels_ = labels

        self.iterations_ = iteration + 1




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

    raise ValueError(
        f"Unsupported file type: {os.path.splitext(file_path)[1]}")


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
        raise ValueError(
            f"Ungültiger Spaltenindex. Das DataFrame hat nur {len(data_frame.columns)} Spalten.")

    selected_columns = [data_frame.columns[idx] for idx in columns]
    return data_frame[selected_columns]


def perform_clustering(data_frame: pd.DataFrame, n_clusters: int, distance_metric: str = "EUCLIDEAN") -> dict:
    kmeans = CustomKMeans(n_clusters=n_clusters, distance_metric=distance_metric)
    kmeans.fit(data_frame.values)
    
    clusters = []
    for idx in range(n_clusters):
        cluster_points = [{"x": point[0], "y": point[1]}
                         for point, label in zip(data_frame.values, kmeans.labels_) if label == idx]
        centroid = {"x": kmeans.cluster_centers_[idx][0], "y": kmeans.cluster_centers_[idx][1]}
        clusters.append({"clusterNr": idx, "centroid": centroid, "points": cluster_points})

    response_data = {
        "name": "K-Means Clustering Ergebnis",
        "cluster": clusters,
        "x_label": data_frame.columns[0],
        "y_label": data_frame.columns[1],
        "distance_metric": distance_metric,
        "iterations": kmeans.iterations_
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
