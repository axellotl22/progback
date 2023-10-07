"""
basic_kmeans_model.py
---------------------
Contains the basic data models used for the KMeans clustering process.
"""

from typing import List, Dict
from pydantic import BaseModel


class Centroid(BaseModel):
    """
    Model representing a centroid of a cluster in 2D.

    Attributes:
    - x (float): X-coordinate of the centroid.
    - y (float): Y-coordinate of the centroid.
    """
    x: float
    y: float


class Centroid3D(BaseModel):
    """
    Model representing a centroid of a cluster in 3D.

    Attributes:
    - x (float): X-coordinate of the centroid.
    - y (float): Y-coordinate of the centroid.
    - z (float): Z-coordinate of the centroid.
    """
    x: float
    y: float
    z: float


class Cluster(BaseModel):
    """
    Model representing a single cluster in 2D.

    Attributes:
    - cluster_nr (int): Number representing the cluster.
    - centroid (Centroid): The centroid of the cluster.
    - points (List[Dict[str, float]]): List of points within the cluster.
    """
    clusterNr: int
    centroid: Centroid
    points: List[Dict[str, float]]


class Cluster3D(Cluster):
    """
    Model representing a single cluster in 3D.

    Attributes:
    - centroid (Centroid3D): The centroid of the cluster.
    """
    centroid: Centroid3D


class BasicKMeansResult(BaseModel):
    """
    Model representing the result of the KMeans clustering process in 2D.

    Attributes:
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - clusters (List[Cluster]): List of resulting clusters.
    - x_label (str): Label for the X-coordinate.
    - y_label (str): Label for the Y-coordinate.
    - iterations (int): Number of iterations the algorithm ran.
    - used_distance_metric (str): The distance metric used for clustering.
    - filename (str): Name of the file containing the data points.
    - k_value (int): Number of clusters used.
    """
    user_id: int
    request_id: int
    cluster: List[Cluster]
    x_label: str
    y_label: str
    iterations: int
    used_distance_metric: str
    name: str
    k_value: int


class KMeansResult3D(BasicKMeansResult):
    """
    Model representing the result of the KMeans clustering process in 3D.

    Attributes:
    - z_label (str): Label for the Z-coordinate.
    """
    z_label: str
    cluster: List[Cluster3D]


class KMeansResultND(BaseModel):
    """
    Model representing the result of the KMeans clustering process in n dimensions.

    Attributes:
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - clusters (List[Cluster]): List of resulting clusters.
    - x_label (str): Label for the X-coordinate.
    - y_label (str): Label for the Y-coordinate.
    - iterations (int): Number of iterations the algorithm ran.
    - used_distance_metric (str): The distance metric used for clustering.
    - filename (str): Name of the file containing the data points.
    - k_value (int): Number of clusters used.
    - important_features (Dict[str, float]): Dictionary of important features with their contributions.
    """
    user_id: int
    request_id: int
    clusters: List[Cluster]
    x_label: str
    y_label: str
    iterations: int
    used_distance_metric: str
    name: str
    k_value: int
    important_features: Dict[str, float]