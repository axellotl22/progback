""" Modelle für Classification. """

from typing import List, Optional, Union
from pydantic import BaseModel
from enum import Enum

class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload.
    
    Attributes:
    - filename (str): Name der hochgeladenen Datei.
    """
    filename: str

class SplitStrategy(Enum):
    BEST_SPLIT = "Best Split"
    MEDIAN = "Median"
    DURCHSCHNITT = "Durchschnitt" 
    RANDOM_SPLIT = "Random Split"
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
    - min_samples_split (int): Mindestanzahl an Dateneinträgen, die an Knoten vorhanden sein muss, für weiteren Split
    - max_depth (int): Maximale Tiefe/Anzahl an Ebenen, die der Baum haben soll
    - features_count (int): Anzahl zu betrachtender Features
    - className: Klassenüberschrift
    - feature_weights: Gewichtung der einzelnen Features für Split
    - split_strategy: Verfahren, nachdem gesplittet werden soll
    """
    root: TreeNode
    min_samples_split: int
    max_depth: int
    features_count: Optional[int]
    className: Optional[str]
    feature_weights: Optional[List[int]]
    split_strategy: Optional[SplitStrategy]
        
    

