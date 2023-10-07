"""
Services for clustering functions. 
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .clustering_algorithms import CustomKMeans
from .utils import clean_dataframe, select_columns
from .elbow_service import run_standard_elbow_method

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_elbow(file_path):
    """
    Determines optimal number of clusters using elbow.
    """
    results = run_standard_elbow_method(file_path=file_path)
    optimal_clusters = int(results.recommended_point.x)
    return optimal_clusters + 1
    
    

def calculate_silhouette(data_frame, max_clusters):
    """
    Determines optimal number of clusters using silhouette score.
    """

    # Calculate silhouette scores for different k
    silhouette_scores = [silhouette_score(data_frame,
                                          KMeans(n_clusters=i,
                                                 init='k-means++',
                                                 max_iter=300,
                                                 n_init=10,
                                                 random_state=0).fit(data_frame).labels_)
                         for i in range(2, max_clusters+1)]

    # Optimal k is where silhouette score is maximum
    # Add 2 because we start calculating scores at k=2
    return np.argmax(silhouette_scores) + 2


def perform_clustering(data_frame, num_clusters, distance_metric="EUCLIDEAN"):
    """
    Performs clustering and returns results.
    """

    # Create CustomKMeans model
    kmeans = CustomKMeans(num_clusters)

    # Fit model to data
    kmeans.fit(data_frame.values)

    # Get cluster labels
    labels = kmeans.labels_

    # Create cluster results
    clusters = [
        {
            "clusterNr": idx,
            "centroid": {"x": centroid[0], "y": centroid[1]},
            "points": [{"x": point[0], "y": point[1]} for point,
                       label in zip(data_frame.values, labels) if label == idx]
        }
        for idx, centroid in enumerate(kmeans.cluster_centers_)
    ]

    # Create results dictionary
    results = {
        "name": "K-Means Clustering Result",
        "cluster": clusters,
        "x_label": data_frame.columns[0],
        "y_label": data_frame.columns[1],
        "iterations": kmeans.iterations_,
        "distance_metric": distance_metric
    }

    return results

# pylint: disable=too-many-arguments
def process_and_cluster(data_frame, method="ELBOW", distance_metric="EUCLIDEAN",
                        columns=None, num_clusters=None, file_path=None):
    """
    Processes data frame and performs clustering.

    Args:
    - data_frame (DataFrame): Data for clustering
    - method (str): Method to determine number of clusters
    - distance_metric (str): Distance metric for clustering
    - columns (list): Columns to use for clustering
    - num_clusters (int): Specified number of clusters
    - file_path (str): File path for elbow method (if needed)

    Returns:
    - dict: Results of clustering 
    """
    
    # Clean and select columns
    data_frame = clean_dataframe(data_frame)
    if columns:
        data_frame = select_columns(data_frame, columns)

    # Get max clusters to try
    max_clusters = min(int(0.25 * data_frame.shape[0]), 50)

    # Calculate optimal clusters using both methods
    optimal_clusters_elbow = calculate_elbow(file_path)
    optimal_clusters_silhouette = calculate_silhouette(data_frame, max_clusters)

    # Use provided num_clusters if given, else choose based on specified method
    if num_clusters:
        optimal_clusters = num_clusters
    elif method == "ELBOW":
        optimal_clusters = optimal_clusters_elbow
    elif method == "SILHOUETTE":
        optimal_clusters = optimal_clusters_silhouette
    else:
        raise ValueError("Invalid method provided. Choose either 'ELBOW' or 'SILHOUETTE'.")

    # Perform clustering
    result = perform_clustering(data_frame, optimal_clusters, distance_metric)

    # Add optimal cluster values to result
    result["clusters_elbow"] = optimal_clusters_elbow
    result["clusters_silhouette"] = optimal_clusters_silhouette

    return result
