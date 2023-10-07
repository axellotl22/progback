"""Modelle für Decision Tree Classification"""

from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel

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
    
class BestSplitStrategy(Enum):
    """
    Enum zur Auswahl des Verfahrens, was der BestSplit ist
    """
    GINI = "Gini-Index"
    ENTROPY = "Entropy"
    INFORMATION_GAIN = "Information Gain"
    NONE = "None"

class LeaveNode(BaseModel):
    """
    Modell eines Blattknotens im Decision Tree
    """
    value: Union[int, None] 

class FeatureNode(BaseModel):
    """
    Modell für Root-Node und Feature/Splitnodes im Decision Tree
    Knoten hat eine Bedingung, sowie Kindsknoten
    """
    feature_name: str
    feature_id: int
    feature_id_name: Optional[str]
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
    - min_samples_split (int): Mindestanzahl an Einträgen für Split
    - max_depth (int): Maximale Tiefe/Anzahl an Ebenen, die der Baum haben soll
    - features_count (int): Anzahl zu betrachtender Features
    - className: Klassenüberschrift
    - feature_weights: Gewichtung der einzelnen Features für Split
    - split_strategy: Verfahren, nachdem gesplittet werden soll
   
    """
    #Tree
    root: FeatureNode
    #Vorhersage und Genauigkeit
    labelclassname: Optional[str]
    self_predictions: Optional[List[str]]
    accuracy: float
    confusion_matrix: Optional[List[List[int]]]
    feature_names: Optional[List[str]]
    #Split Kriterien
    min_samples_split: int
    max_depth: int
    random_state: Optional[int]
    test_size: Optional[float]
    features_count: Optional[int]
    split_strategy: Optional[SplitStrategy]
    best_split_strategy: Optional[BestSplitStrategy]
    feature_behaviour: Optional[bool]
    feature_weights: Optional[List[int]]
    #Stutzen
    pruning: Optional[bool]
    
class DecisionTreeResult(BaseModel):
    """
    Modell für Decision Tree Vorhersagen
    """
    user_id: int
    request_id: int
    name: str
    decision_tree: DecisionTree
    predicted_class: str
    
    