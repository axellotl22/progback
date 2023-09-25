""" Dienste für Clustering-Funktionen. """

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from .clustering_algorithms import CustomKMeans
from .utils import clean_dataframe, select_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_elbow(data_frame, max_clusters):
    """
    Bestimmt die optimale Anzahl von Clustern mittels der Elbow-Methode.

    Args:
    - data_frame (pd.DataFrame): Daten für die Clusteranalyse.
    - max_clusters (int): Maximale Anzahl an Clustern.

    Returns:
    - int: Optimale Anzahl an Clustern.
    """
    wcss = [KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(data_frame).inertia_
            for i in range(1, max_clusters+1)]
    
    differences = np.diff(wcss, n=2)
    return np.argmin(differences) + 3

def calculate_silhouette(data_frame, max_clusters):
    """
    Bestimmt die optimale Anzahl von Clustern mittels des Silhouette-Scores.

    Args:
    - data_frame (pd.DataFrame): Daten für die Clusteranalyse.
    - max_clusters (int): Maximale Anzahl an Clustern.

    Returns:
    - int: Optimale Anzahl an Clustern.
    """
    sil_scores = [silhouette_score(data_frame, KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                                                      n_init=10, random_state=0).fit(data_frame).labels_)
                  for i in range(2, max_clusters+1)]
    return np.argmax(sil_scores) + 2


def perform_clustering(data_frame, n_clusters, distance_metric="EUCLIDEAN"):
    """
    Führt das eigentliche Clustering durch und gibt die Ergebnisse zurück.

    Args:
    - data_frame (pd.DataFrame): Daten für die Clusteranalyse.
    - n_clusters (int): Anzahl von Clustern.
    - distance_metric (str): Distanzmetrik für das Clustering.

    Returns:
    - dict: Ergebnisse des Clusterings.
    """
    kmeans = CustomKMeans(n_clusters=n_clusters, distance_metric=distance_metric)
    kmeans.fit(data_frame.values)
    labels = kmeans.labels_
    
    clusters = [
        {
            "clusterNr": idx,
            "centroid": {"x": centroid[0], "y": centroid[1]},
            "points": [{"x": point[0], "y": point[1]} for point, label in zip(data_frame.values, labels) if label == idx]
        }
        for idx, centroid in enumerate(kmeans.cluster_centers_)
    ]

    return {
        "name": "K-Means Clustering Ergebnis",
        "cluster": clusters,
        "x_label": data_frame.columns[0],
        "y_label": data_frame.columns[1],
        "iterations": kmeans.iterations_,
        "distance_metric": distance_metric
    }

def process_and_cluster(data_frame, method="ELBOW", distance_metric="EUCLIDEAN", columns=None, k_cluster=None):
    """
    Verarbeitet den gegebenen DataFrame und führt das Clustering durch.

    Args:
    - data_frame (pd.DataFrame): Daten für die Clusteranalyse.
    - method (str): Methode zur Bestimmung der optimalen Clusteranzahl.
    - distance_metric (str): Distanzmetrik für das Clustering.
    - columns (list): Zu berücksichtigende Spalten im DataFrame.
    - k_cluster (int, optional): Manuell festgelegte Anzahl von Clustern.

    Returns:
    - dict: Ergebnisse des Clusterings.
    """
    data_frame = clean_dataframe(data_frame)
    if columns:
        data_frame = select_columns(data_frame, columns)
    
    max_clusters = min(int(0.25 * data_frame.shape[0]), 50)
    
    # Berechnen Sie die optimalen Clusterzahlen für beide Methoden
    methods = [calculate_elbow, calculate_silhouette]
    results = Parallel(n_jobs=-1)(delayed(method)(data_frame, max_clusters) for method in methods)
        
    optimal_clusters_methods = {
        "ELBOW": results[0],
        "SILHOUETTE": results[1]
    }
    
    # Wenn k_cluster angegeben ist, verwenden Sie diesen Wert; ansonsten verwenden Sie den Wert aus `optimal_clusters_methods`
    optimal_clusters = k_cluster if k_cluster else optimal_clusters_methods[method]

    result = perform_clustering(data_frame, optimal_clusters, distance_metric)
    result["clusters_elbow"] = results[0]
    result["clusters_silhouette"] = results[1]

    return result