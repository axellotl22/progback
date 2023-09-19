"""
Dienste für Clustering-Funktionen.
"""

import logging

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

from .clustering_algorithms import CustomKMeans

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

def determine_optimal_clusters(data_frame) -> int:
    """
    Bestimmt die optimale Clusteranzahl mittels der Kombination von Silhouetten-Methode und Davies-Bouldin Index.
    """
    sil_scores = []
    dbi_scores = []
    max_clusters = data_frame.shape[0] - 1

    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(data_frame)
        labels = kmeans.labels_

        # Silhouettenwert berechnen
        sil_scores.append(silhouette_score(data_frame, labels))
        
        # Davies-Bouldin Index berechnen
        dbi_scores.append(davies_bouldin_score(data_frame, labels))

    # Den Silhouettenwert normalisieren
    sil_scores = [score/max(sil_scores) for score in sil_scores]
    
    # DBI invertieren und normalisieren, da ein niedrigerer DBI besser ist
    dbi_scores = [1 - (score/max(dbi_scores)) for score in dbi_scores]

    # Eine kombinierte Bewertung berechnen
    combined_scores = [sil + dbi for sil, dbi in zip(sil_scores, dbi_scores)]

    return list(range(2, max_clusters))[combined_scores.index(max(combined_scores))]

def perform_clustering(data_frame, n_clusters: int, distance_metric: str = "EUCLIDEAN") -> dict:
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
