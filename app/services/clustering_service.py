""" Dienste für Clustering-Funktionen. """

import logging
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from .clustering_algorithms import CustomKMeans
from .utils import clean_dataframe, select_columns

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cluster_and_score(i, data_frame):
    """Hilfsfunktion zur Berechnung von Silhouetten- 
    und Davies-Bouldin-Werten für einen gegebenen Cluster."""

    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    max_iter=300, n_init=10,
                    random_state=0).fit(data_frame)
    labels = kmeans.labels_

    sil_score = silhouette_score(data_frame, labels)
    dbi_score = davies_bouldin_score(data_frame, labels)

    return sil_score, dbi_score


def determine_optimal_clusters(data_frame):
    """Bestimmt die optimale Clusteranzahl."""

    logger.info("Starting to determine the optimal number of clusters.")

    max_clusters = min(int(0.2 * data_frame.shape[0]), 30)
    logger.info("Max clusters set to: %s", max_clusters)

    results = Parallel(n_jobs=-1)(delayed(cluster_and_score)
                                  (i, data_frame) for i in range(2, max_clusters))
    sil_scores, dbi_scores = zip(*results)

    sil_scores = [score/max(sil_scores) for score in sil_scores]
    dbi_scores = [1 - (score/max(dbi_scores)) for score in dbi_scores]
    combined_scores = [sil + dbi for sil, dbi in zip(sil_scores, dbi_scores)]
    optimal_clusters = list(range(2, max_clusters))[
        combined_scores.index(max(combined_scores))]

    logger.info("Optimal number of clusters determined as: %s",
                optimal_clusters)

    return optimal_clusters


def perform_clustering(data_frame, n_clusters, distance_metric="EUCLIDEAN"):
    """Führt das Clustering mit gegebenen Parametern durch und gibt die Ergebnisse zurück."""

    logger.info("Starting clustering with %s clusters and %s distance metric.",
                n_clusters, distance_metric)

    kmeans = CustomKMeans(n_clusters=n_clusters,
                          distance_metric=distance_metric)
    kmeans.fit(data_frame.values)

    clusters = []
    for idx in range(n_clusters):
        cluster_points = [{"x": point[0], "y": point[1]}
                          for point, label in zip(data_frame.values, kmeans.labels_)
                          if label == idx]
        centroid = {"x": kmeans.cluster_centers_[
            idx][0], "y": kmeans.cluster_centers_[idx][1]}
        clusters.append(
            {"clusterNr": idx, "centroid": centroid, "points": cluster_points})

    response_data = {
        "name": "K-Means Clustering Ergebnis",
        "cluster": clusters,
        "x_label": data_frame.columns[0],
        "y_label": data_frame.columns[1],
        "distance_metric": distance_metric,
        "iterations": kmeans.iterations_
    }

    logger.info("Clustering completed successfully.")

    return response_data

def process_and_cluster(data_frame, clusters, distance_metric, columns=None):
    """Verarbeitet und clustert die Daten und gibt das Ergebnis zurück."""
    data_frame = clean_dataframe(data_frame)
    if columns:
        data_frame = select_columns(data_frame, columns)

    if clusters is None:
        optimal_clusters = determine_optimal_clusters(data_frame)
    else:
        if clusters <= 1 or clusters > len(data_frame):
            raise ValueError("Ungültige Anzahl von Clustern")
        optimal_clusters = clusters

    return perform_clustering(data_frame, optimal_clusters, distance_metric)
