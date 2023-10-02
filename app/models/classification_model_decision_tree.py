"""Modelle für Decision Tree Classification"""

from typing import List, Optional, Union
from pydantic import BaseModel
from enum import Enum

class FileUpload(BaseModel):
    """
    Modell für den Datei-Upload
    
    Attribute:
    - filename (str): Name der hochgeladenen Datei
    """
    filename: str

class SplitStrategy(Enum):
    """
    Enum zur Auswahl der Split-Strategie
    """
    BEST_SPLIT = "Best Split"
    MEDIAN = "Median"
    DURCHSCHNITT = "Durchschnitt" 
    RANDOM_SPLIT = "Random Split"
    
class BestSplitStrategy:
    """
    Enum zur Auswahl des Verfahrens, was der BestSplit ist
    """
    GINI = "Gini-Index"
    ENTROPY = "Entropy"
    INFORMATION_GAIN = "Information Gain"
    
class Node(BaseModel):
    """
    Modell für einzelne Nodes des Decision Trees
    
    Attribute:
    - feature_id: ID des Features
    - treshold: Richtwert des Knotens
    - left: Linker Kindknoten
    - right: rechter Kindknoten
    - value: Klasse, falls Leave Node
    - feature_name: Name des Features
    """
    feature_id: int
    treshold: float
    left: Union['Node', None]
    right: Union['Node', None]
    feature_name: str
Node.model_rebuild()

class LeaveNode(BaseModel):
    value: Union[int, None] 

class FeatureNode(BaseModel):
    feature_name: str
    feature_id: int
    left: Union['FeatureNode','LeaveNode', None]
    right: Union['FeatureNode','LeaveNode', None]
    treshold: float

class DecisionTreeTrainingsData(BaseModel):
    """
    Modell für die Trainingsdaten des Decision Trees
    
    Attribute:
    - X (List[List[float]]): Feature-Datenmatrix.
    - y (List[int]): Zielvariable.
    """
    features: List[List[float]]
    labels: List[str]
    
class DecisionTree(BaseModel):
    """
    Modell für Decision Tree Struktur und Erstellungskriterien
    
    Attribute:
    - root (TreeNode): Wurzelknoten des Entscheidungsbaums.
    - min_samples_split (int): Mindestanzahl an Dateneinträgen, die an Knoten vorhanden sein muss, für weiteren Split
    - max_depth (int): Maximale Tiefe/Anzahl an Ebenen, die der Baum haben soll
    - features_count (int): Anzahl zu betrachtender Features
    - className: Klassenüberschrift
    - feature_weights: Gewichtung der einzelnen Features für Split
    - split_strategy: Verfahren, nachdem gesplittet werden soll
   
    """
    root: FeatureNode
    min_samples_split: int
    max_depth: int
    features_count: Optional[int]
    labelclassname: Optional[str]
    #feature_weights: Optional[List[int]]
    split_strategy: Optional[SplitStrategy]
    
class DecisionTreeResult(BaseModel):
    """
    Modell für Decision Tree Vorhersagen
    """
    user_id: int
    request_id: int
    name: str
    decision_tree: DecisionTree
    predicted_class: str