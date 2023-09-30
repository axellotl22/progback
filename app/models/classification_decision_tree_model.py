""" Modelle für Classification. """

from typing import List, Optional, Union
from pydantic import BaseModel

class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload.
    
    Attributes:
    - filename (str): Name der hochgeladenen Datei.
    """
    filename: str


class TreeNode(BaseModel):
    """
    Modell für einen Knoten im Entscheidungsbaum.
    
    Attributes:
    - feature_id (Optional[int]): ID des Features.
    - treshold (Optional[float]): Schwellenwert für den Knoten.
    - left (Optional[Union['TreeNode', None]]): Linker Kindknoten.
    - right (Optional[Union['TreeNode', None]]): Rechter Kindknoten.
    - value (Optional[int]): Wert des Knotens, wenn es sich um einen Blattknoten handelt.
    - feature_name (Optional[str]): Name des Features.
    """
    feature_id: int
    treshold: float
    left: Union['TreeNode', None]
    right: Union['TreeNode', None]
    value: int
    feature_name: str
TreeNode.model_rebuild()

class DecisionTreeData(BaseModel):
    """
    Modell für die Trainingsdaten des Entscheidungsbaums.
    
    Attributes:
    - X (List[List[float]]): Feature-Datenmatrix.
    - y (List[int]): Zielvariable.
    """
    features: List[List[float]]
    labels: List[str]

class DecisionTreeResult(BaseModel):
    """
    Modell für das Ergebnis des Entscheidungsbaum-Prozesses.
    
    Attributes:
    - root (TreeNode): Wurzelknoten des Entscheidungsbaums.
    """
    root: TreeNode