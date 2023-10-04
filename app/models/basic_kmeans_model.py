"""
basic_kmeans_model.py
---------------------
Contains the basic data models used for the KMeans clustering process.
"""

from typing import List, Dict
from pydantic import BaseModel


class Centroid(BaseModel):
    """
    Model representing a centroid of a cluster.

    Attributes:
    - x (float): X-coordinate of the centroid.
    - y (float): Y-coordinate of the centroid.
    """

    x: float
    y: float


class Cluster(BaseModel):
    """
    Model representing a single cluster.

    Attributes:
    - clusterNr (int): Number representing the cluster.
    - centroid (Centroid): The centroid of the cluster.
    - points (List[Dict[str, float]]): List of points within the cluster.
    """

    clusterNr: int
    centroid: Centroid
    points: List[Dict[str, float]]


class KMeansResult(BaseModel):
    """
    Model representing the result of the KMeans clustering process.

    Attributes:
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - clusters (List[Cluster]): List of resulting clusters.
    - iterations (int): Number of iterations the algorithm ran.
    - used_distance_metric (str): The distance metric used for clustering.
    - filename (str): Name of the file containing the data points.
    - k_value (int): Number of clusters used.
    """

    user_id: int
    request_id: int
    clusters: List[Cluster]
    x_label:str
    y_label:str
    iterations: int
    used_distance_metric: str
    filename: str
    k_value: int
