"""
basic_kmeans_model.py
---------------------
Module that defines data models used in the KMeans clustering process.
"""

from typing import List, Dict
from pydantic import BaseModel


class Centroid(BaseModel):
    """
    Represents a 2D centroid for KMeans clustering.

    Attributes:
        x (float): Horizontal position of the centroid.
        y (float): Vertical position of the centroid.
    """
    x: float
    y: float


class Centroid3D(BaseModel):
    """
    Represents a 3D centroid for KMeans clustering.

    Attributes:
        x (float): Horizontal position of the centroid.
        y (float): Vertical position of the centroid.
        z (float): Depth position of the centroid.
    """
    x: float
    y: float
    z: float


class Cluster(BaseModel):
    """
    Describes a 2D cluster for KMeans clustering.

    Attributes:
        clusterNr (int): Unique identifier for the cluster.
        centroid (Centroid): Geometric center of the cluster.
        points (List[Dict[str, float]]): Collection of data points associated with this cluster.
    """
    clusterNr: int
    centroid: Centroid
    points: List[Dict[str, float]]


class Cluster3D(Cluster):
    """
    Describes a 3D cluster for KMeans clustering.

    Attributes:
        centroid (Centroid3D): Geometric center of the cluster.
    """
    centroid: Centroid3D


class BasicKMeansResult(BaseModel):
    """
    Encapsulates the results of a 2D KMeans clustering operation.

    Attributes:
        user_id (int): Identifier for the user initiating the request.
        request_id (int): Unique identifier for the clustering request.
        cluster (List[Cluster]): Collection of identified clusters.
        x_label (str): Description for the horizontal axis.
        y_label (str): Description for the vertical axis.
        iterations (int): Total number of iterations executed by the algorithm.
        used_distance_metric (str): Metric used to determine distances between data points.
        name (str): Reference name for the data source.
        k_value (int): Specified number of clusters for the operation.
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
    Encapsulates the results of a 3D KMeans clustering operation.

    Attributes:
        z_label (str): Description for the depth axis.
        cluster (List[Cluster3D]): Collection of identified 3D clusters.
    """
    z_label: str
    cluster: List[Cluster3D]
