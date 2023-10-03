"""
basic_kmeans_service.py
-----------------------
Service for performing KMeans clustering using the optimized KMeans and MiniBatch KMeans.
"""

from typing import Union
import numpy as np
from .custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import KMeansInput, KMeansResult, Cluster, Centroid  
from app.services.utils import load_and_process_data

def perform_kmeans_clustering(
    input_data: KMeansInput, 
    filename: str, 
    user_id: int, 
    request_id: int, 
    column1: Union[str, int], 
    column2: Union[str, int], 
    method: str = "optimized", 
    distance_metric: str = "EUCLIDEAN"
) -> KMeansResult:

    # Load data with the updated function
    data_points, x_label, y_label = load_and_process_data(filename, column1, column2)
    
    if not input_data.k:
        raise ValueError("'k' must be specified for clustering.")  

    # Choose the KMeans algorithm
    if method == "optimized":
        kmeans = OptimizedKMeans(n_clusters=input_data.k, distance_metric=distance_metric)
    elif method == "minibatch":
        kmeans = OptimizedMiniBatchKMeans(n_clusters=input_data.k, distance_metric=distance_metric)
    else:
        raise ValueError("Invalid KMeans method. Choose 'optimized' or 'minibatch'.")

    kmeans.fit(data_points)
    
    labels = np.array(kmeans._assign_labels(data_points))
    
    clusters = []
    for label in np.unique(labels):
        points_in_cluster = data_points[labels == label]
        centroid = Centroid(x=kmeans.cluster_centers_[label][0], y=kmeans.cluster_centers_[label][1])
        cluster_points = [{"x": point[0], "y": point[1]} for point in points_in_cluster]
        clusters.append(Cluster(clusterNr=label, centroid=centroid, points=cluster_points))

    result = KMeansResult(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=kmeans.max_iterations,
        used_distance_metric=kmeans.distance_metric,
        filename=filename,
        k_value=input_data.k
    )

    return result
