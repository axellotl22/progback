"""Modelle für Clustering."""

from typing import List, Dict
from pydantic import BaseModel


class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload.

    Attribute:
    - filename (str): Name der hochgeladenen Datei
    """

    filename: str


class Centroid(BaseModel):
    """
    Modell für ein Zentroid eines Clusters.

    Attribute:  
    - x (float): X-Koordinate des Zentroiden
    - y (float): Y-Koordinate des Zentroiden
    """

    x: float
    y: float


class Cluster(BaseModel):
    """
    Modell für ein Cluster.

    Attribute:
    - clusterNr (int): Nummer des Clusters
    - centroid (Centroid): Zentroid des Clusters
    - points (List[Dict[str, float]]): Liste von Punkten innerhalb des Clusters
    """

    clusterNr: int
    centroid: Centroid
    points: List[Dict[str, float]]

# pylint: disable=R0801
class ClusterResult(BaseModel):
    """
    Modell für das Ergebnis des Clustering-Prozesses.

    Attribute:
    - user_id (int): User-ID
    - request_id (int): Request-ID
    - name (str): Name des Clustering-Ergebnisses
    - cluster (List[Cluster]): Liste von Clustern  
    - x_label (str): Label für die X-Achse
    - y_label (str): Label für die Y-Achse
    - iterations (int): Anzahl der Iterationen des Clustering-Algorithmus
    - used_distance_metric (str): Verwendetes Distanzmaß
    - used_optK_method (str): Verwendete Methode zur Bestimmung von k
    - clusters_elbow (int): Anzahl Cluster (Elbow-Methode)
    - clusters_silhouette (int): Anzahl Cluster (Silhouette-Methode) 
    """

    user_id: int
    request_id: int
    name: str
    cluster: List[Cluster]
    x_label: str
    y_label: str
    iterations: int
    used_distance_metric: str
    used_optK_method: str
    clusters_elbow: int
    clusters_silhouette: int
