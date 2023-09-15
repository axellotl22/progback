""" Modelle für Clustering. """

from typing import List, Dict

from pydantic import BaseModel

class FileUpload(BaseModel):
    filename: str

class ClusterPoint(BaseModel):
    x: float 
    y: float
    cluster: int

class ClusterResult(BaseModel):
    points: List[ClusterPoint] 
    centroids: List[ClusterPoint]
    point_to_centroid_mappings: Dict[int, int]