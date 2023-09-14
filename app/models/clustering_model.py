"""
Modelle für Clustering.
"""

from typing import List

from pydantic import BaseModel

class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload.
    """
    filename: str

class ClusterResult(BaseModel):
    """
    Modell für das Ergebnis des Clusterings.
    """
    cluster_labels: List[int]
    optimal_cluster_count: int
