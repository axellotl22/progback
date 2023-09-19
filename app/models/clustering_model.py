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

class Centroid(BaseModel):
    """
    Modell für ein Zentroid eines Clusters.
    
    Attributes:
    - x (float): X-Koordinate des Zentroiden.
    - y (float): Y-Koordinate des Zentroiden.
    """
    x: float
    y: float

class Cluster(BaseModel):
    """
    Modell für ein Cluster.
    
    Attributes:
    - clusterNr (int): Nummer des Clusters.
    - centroid (Centroid): Zentroid des Clusters.
    - points (List[Dict[str, float]]): Liste von Punkten innerhalb des Clusters.
    """
    clusterNr: int
    centroid: Centroid
    points: List[Dict[str, float]]

class ClusterResult(BaseModel):
    """
    Modell für das Ergebnis des Clustering-Prozesses.
    
    Attributes:
    - name (str): Name des Clustering-Ergebnisses.
    - cluster (List[Cluster]): Liste von Clustern.
    - x_label (str): Label für die X-Achse.
    - y_label (str): Label für die Y-Achse.
    """
    
    name: str
    cluster: List[Cluster]
    x_label: str
    y_label: str
