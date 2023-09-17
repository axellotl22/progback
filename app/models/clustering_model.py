""" Modelle für Clustering. """

from typing import List, Dict
from pydantic import BaseModel

class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload.
    
    Attributes:
    - filename (str): Name der hochgeladenen Datei.
    """
    filename: str

class ClusterPoint(BaseModel):
    """
    Modell für einen Punkt innerhalb eines Clusters.
    
    Attributes:
    - x (float): X-Koordinate des Punktes.
    - y (float): Y-Koordinate des Punktes.
    - cluster (int): Zugehöriger Cluster des Punktes.
    """
    x: float
    y: float
    cluster: int

class ClusterResult(BaseModel):
    """
    Modell für das Ergebnis des Clustering-Prozesses.
    
    Attributes:
    - points (List[ClusterPoint]): Liste von Punkten innerhalb der Cluster.
    - centroids (List[ClusterPoint]): Liste von Zentroiden für die Cluster.
    - point_to_centroid_mappings (Dict[int, int]): Zuordnung von Punkten zu ihren Zentroiden.
    """
    points: List[ClusterPoint]
    centroids: List[ClusterPoint]
    point_to_centroid_mappings: Dict[int, int]
    x_label: str
    y_label: str
