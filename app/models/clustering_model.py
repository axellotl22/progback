from pydantic import BaseModel
from typing import List

class FileUpload(BaseModel):
    filename: str

class ClusterResult(BaseModel):
    cluster_labels: List[int]
    optimal_cluster_count: int