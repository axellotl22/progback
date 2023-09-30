""" Modelle für Classification. """

"""
PArameter Customizing
Best Split Verfahren (Gini, Entropy), Best Split vs Random Split
Mit/Ohne Zurücklegen der Features
Max Depth
Anzahl zu verwendender Features
Mindestanzahl an Daten für Split
Feature Gewichtung
Stutzen
Vorsortieren der Daten
Splitter STrategie (Best, Median, Durchschnitt)
Confusion MAtrix
"""
import numpy as np
from collections import Counter
from enum import Enum
import pandas as pd


class SplitStrategy(Enum):
    BEST_SPLIT = "Best Split"
    MEDIAN = "Median"
    DURCHSCHNITT = "Durchschnitt" 
    RANDOM_SPLIT = "Random Split"

class Node:
    def __init__(self, feature_id=None, treshold=None, left=None, right=None,*,value=None):
        self.feature_id = feature_id
        self.treshold = treshold
        self.left = left
        self.right= right
        self.value= value
        
       
        
    def is_leave(self):
        return self.value is not None
        
        
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, features_count=None, features_names=None, class_Name=None, feature_weights=None, split_strategy= None):
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.features_count = features_count
        self.features_names = features_names
        self.class_Name = class_Name
        self.feature_weights = feature_weights if feature_weights is not None else np.ones(features_count)  
        self.split_strategy = split_strategy 
        self.root=None
        
    def fit(self, X, y):
        self.features_count = X.shape[1] if not self.features_count else  min(X.shape[1], self.features_count)
        self.root = self.create_tree(X, y)
    
    def create_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        #Abbruchkriterium
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self.most_frequent_label(y)
            return Node(value=leaf_value)
        
        features = np.random.choice(n_feats, self.features_count, replace=False)
        #Best Split ermitteln
        
        best_feature, best_treshold = self.best_split(X, y, features)
        #Aufbauen des Baums
        id_left, id_right = self.split(X[:, best_feature], best_treshold)
        left = self.create_tree(X[id_left, :], y[id_left], depth+1)
        right = self.create_tree(X[id_right, :], y[id_right], depth+1)
        return Node(best_feature, best_treshold, left, right)
    
    def extract_feature_names(df, target_column):
        """
        Extrahiert die Feature-Namen aus einem DataFrame.

        Parameters:
        - df: pandas DataFrame
        - target_column: Name der Ziel-Spalte (z.B. "Drug")

        Returns:
        - Liste der Feature-Namen
        """
        return [col for col in df.columns if col != target_column]
    def confusion_matrix(self, y_true, y_pred):
        """
        Erstellt eine Confusion Matrix für binäre Klassifikation.

        Parameters:
        - y_true: Die wahren Klassenlabels.
        - y_pred: Die vorhergesagten Klassenlabels.

        Returns:
        - Eine 2x2 Confusion Matrix.
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
    
        return np.array([[TP, FP], [FN, TN]])
    
    def best_split (self, X, y, features):
        best_gain = -1
        split_number, split_treshold = None, None 
        
        for feat in features:
            X_column = X[:, feat]
            #Wenn Best Split ermittelt
            tresholds = np.unique (X_column)
            
            for treshold in tresholds:
                information_gain=self.calc_information_gain(y, X_column, treshold)
                
                if (information_gain > best_gain):
                    best_gain = information_gain
                    split_number = feat
                    split_treshold = treshold
              
        return split_number, split_treshold
          
    
    
      
     
    def choose_split(self, X, y, features, strategy):
        best_gain = -1
        split_number, split_treshold = None, None 

        for feat in features:
            X_column = X[:, feat]
            
            if strategy == SplitStrategy.MEDIAN:
                treshold = np.median(X_column)
            elif strategy == SplitStrategy.DURCHSCHNITT:
                treshold = np.mean(X_column)
            elif strategy == SplitStrategy.RANDOM_SPLIT:
                treshold = np.random.choice(X_column)
            else:  # default to "best"
                tresholds = np.unique(X_column)
                for treshold in tresholds:
                    gain = self.calc_information_gain(y, X_column, treshold)
                    weighted_gain=self.feature_weights*gain
                    if weighted_gain > best_gain:
                        best_gain = weighted_gain
                        split_number = feat
                        split_treshold = treshold
                continue  # Skip the rest of the loop for "best" strategy

            gain = self.calc_information_gain(y, X_column, treshold)
            if gain > best_gain:
                best_gain = gain
                split_number = feat
                split_treshold = treshold

        return split_number, split_treshold          
          
            
               
            
    def accuracy (y_test, y_pred):
        return np.sum(y_test==y_pred)/len(y_test)
        
    def calc_information_gain(self, y, X_column, treshold):
        #IG=E(parent)-[weighted averege]*E(children)
        #parent entropy
        parent_entropy = self.calc_entropy(y)
        #create children
        id_left, id_right = self.split(X_column, treshold)
        
        if len(id_left) == 0 or len (id_right)==0:
            return 0
        #calculated weighted entropy of children
        n = len(y)
        n_l, n_r = len(id_left), len(id_right)
        e_l, e_r = self.calc_entropy(y[id_left]), self.calc_entropy(y[id_right])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        #calculate Information Gain
        information_gain = parent_entropy-child_entropy
        return information_gain
        
        
    def calc_entropy(self, y):
        # -Sum of p(X)*log2(p(X)), P(X) = Number of x/number of values
        count_numbers = np.bincount(y) #Array how often which number is used
        p_vals = count_numbers/len(y)
        return -np.sum([p*np.log2(p) for p in p_vals if p>0])
    def split (self, X_column, split_treshold):
        id_left = np.argwhere(X_column<=split_treshold).flatten()
        id_right = np.argwhere(X_column>split_treshold).flatten()
        return id_left, id_right
        
    def calc_gini():
        count_numbers = np.bincount(y) #Array how often which number is used
        p_vals = count_numbers/len(y)
        return 1-np.sum([p*p for p in p_vals if p>0])
    
    
    
    #def calc_gini_index():   
    def most_frequent_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def traverse_tree(self, x, node):
        if node.is_leave():
            return node.value
        
        if x[node.feature_id] <= node.treshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root)for x in X])
   
    def to_json(self, node=None):
        if node is None:
            node = self.root

        if node.is_leave():
            return {
                "label": f"Class {node.value}",
                "question": None,
                "children": []
            }

        left_child = self.to_json(node.left)
        right_child = self.to_json(node.right)

        return {
            "label": None,
            "question": f"Is feature {node.feature_id} <= {node.treshold}?",
            "children": [left_child, right_child]
        }

    def sort_data(self, X, y, feature_index):
        """
    Sortiert die Daten basierend auf einem bestimmten Feature.

    Parameters:
    - X: Die Eingabedaten.
    - y: Die Zielwerte.
    - feature_index: Der Index des Features, nach dem sortiert werden soll.

    Returns:
    - X_sorted: Die sortierten Eingabedaten.
    - y_sorted: Die entsprechend sortierten Zielwerte.
    """
        # Sortiert die Indizes von X basierend auf dem gegebenen Feature.
        sorted_indices = np.argsort(X[:, feature_index])

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
    
        return X_sorted, y_sorted

    def node_error(self, y):
        """
        Berechnet den Fehler eines Knotens basierend auf den gegebenen Zielwerten.

        Parameters:
        - y: Die Zielwerte.

        Returns:
        - error: Der Fehler des Knotens.
        """
        most_common_label = self.most_frequent_label(y)
        error = sum([1 for label in y if label != most_common_label])
        return error

    def prune(self, node, X, y):
        """
        Stutzt den Baum rekursiv, um Overfitting zu verhindern.

        Parameters:
        - node: Der aktuelle Knoten.
        - X: Die Eingabedaten.
        - y: Die Zielwerte.
        """
        if node.is_leave():
            return
        id_left, id_right = self.split(X[:, node.feature_id], node.treshold)
        if node.left:
            self.prune(node.left, X[id_left], y[id_left])
        if node.right:
            self.prune(node.right, X[id_right], y[id_right])
        if node.left.is_leave() and node.right.is_leave():
            error_without_prune = self.node_error(y)
            error_with_prune = self.node_error(y[id_left]) + self.node_error(y[id_right])     
            if error_with_prune < error_without_prune:
                node.left = None
                node.right = None
                node.value = self.most_frequent_label(y)
    
    

    