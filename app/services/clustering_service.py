"""
Services for clustering functions. 
"""

import logging
from joblib import Parallel, delayed   
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .clustering_algorithms import CustomKMeans
from .utils import clean_dataframe, select_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_elbow(data_frame, max_clusters):
    """
    Determines optimal number of clusters using elbow method.
    
    Elbow method looks at within cluster sum of squares (WCSS) 
    for different values of k. The optimal k is at the "elbow"
    where the curve bends.
    
    Args:
    - data_frame (DataFrame): Data for clustering
    - max_clusters (int): Maximum number of clusters to check
    
    Returns:
    - int: Optimal number of clusters based on elbow method
    """
    
    # Calculate WCSS for range of clusters
    wcss = [KMeans(n_clusters=i, init='k-means++',
                   max_iter=300, n_init=10,
                   random_state=0).fit(data_frame).inertia_
            for i in range(1, max_clusters+1)]
            
    # Take second order difference of WCSS    
    differences = np.diff(wcss, n=2)
    
    # Optimal k is at minimum of second order differences
    # Add 3 because we took second order differences
    return np.argmin(differences) + 3


def calculate_silhouette(data_frame, max_clusters):
    """
    Determines optimal number of clusters using silhouette score.
    
    Silhouette score measures how well samples are clustered. 
    Score is between -1 and 1, with a higher score indicating
    better clustering.
    
    Args:
    - data_frame (DataFrame): Data for clustering 
    - max_clusters (int): Max number of clusters to check
    
    Returns:
    - int: Optimal number of clusters based on silhouette score
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

    Args:
    - data_frame (DataFrame): Data for clustering
    - num_clusters (int): Number of clusters
    - distance_metric (string): Distance metric to use
    
    Returns:
    - dict: Results of clustering
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


def process_and_cluster(data_frame, method="ELBOW", distance_metric="EUCLIDEAN",
                        columns=None, num_clusters=None):
    """
    Processes data frame and performs clustering.

    Args:
    - data_frame (DataFrame): Data for clustering
    - method (str): Method to determine number of clusters
    - distance_metric (str): Distance metric for clustering
    - columns (list): Columns to use for clustering
    - num_clusters (int): Specified number of clusters

    Returns:
    - dict: Results of clustering 
    """

    # Clean and select columns
    data_frame = clean_dataframe(data_frame)
    if columns:
        data_frame = select_columns(data_frame, columns)

    # Get max clusters to try
    max_clusters = min(int(0.25 * data_frame.shape[0]), 50)

    # Calculate optimal clusters for both methods
    methods = [calculate_elbow, calculate_silhouette]
    results = Parallel(n_jobs=-1)(delayed(method)(data_frame, max_clusters)
                                  for method in methods)

    # Store optimal clusters    
    optimal_clusters_methods = {
        "ELBOW": results[0],
        "SILHOUETTE": results[1]
    }

    # Use provided num_clusters if given
    optimal_clusters = num_clusters if num_clusters else optimal_clusters_methods[method]

    # Perform clustering
    result = perform_clustering(data_frame, optimal_clusters, distance_metric)

    # Add optimal cluster values to result
    result["clusters_elbow"] = results[0]
    result["clusters_silhouette"] = results[1]

    return result
