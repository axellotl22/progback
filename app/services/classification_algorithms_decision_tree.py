"""
Implementierung eines Decision Trees mit Customizing Funktionen:
...
"""
import numpy as np
class CustomNode:
    
    def __init__(self, feature_id=None, treshold=None, left=None, right=None,*,value=None):
        self.feature_id = feature_id
        self.treshold = treshold
        self.left = left
        self.right= right
        self.value= value

class CustomDecisionTree:
    """
    ...
    """
    def __init__(self, min_samples_split=2, max_depth=100, features_count=None, features_names=None, class_Name=None, feature_weights=None, split_strategy= None):
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.features_count = features_count
        self.features_names = features_names
        self.class_Name = class_Name
        self.feature_weights = feature_weights if feature_weights is not None else np.ones(features_count)  
        self.split_strategy = split_strategy 
        self.root=None
    
    
    