"""
Implementierung eines Decision Trees mit Customizing Funktionen:
...
"""
from collections import Counter
import numpy as np
from app.models.classification_model_decision_tree import FeatureNode, LeaveNode, SplitStrategy


class CustomNode:
    """
    Node für Decision Tree
    """
    # pylint: disable=too-many-arguments
    def __init__(self, feature_id=None, treshold=None, left=None, right=None,*,value=None):
        """
        Initialisierung des Custum Nodes
        """
        self.feature_id = feature_id
        self.treshold = treshold
        self.left = left
        self.right= right
        self.value= value
        
    def is_leave(self):
        """
        Ist der Knoten ein Blattknoten?
        Ja -> 1
        Nein ->0
        """
        return self.value is not None
    
    def useless(self):
        """
        Pylint fordert 2 Public Methods
        """
        return None

class CustomDecisionTree:
    """
    DecisionTree Klasse
    """
    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(self, min_samples_split=2, 
                 max_depth=100, 
                 features_count=None, 
                 features_names=None, 
                 class_name=None, 
                 feature_weights=None, 
                 split_strategy= None):
        """
        Initialisierung des Decision Trees
        """
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.features_count = features_count
        self.features_names = features_names
        self.class_name = class_name
        if feature_weights is not None:
            self.feature_weights= feature_weights
        else:
            self.feature_weights= np.ones(features_count)  
        self.split_strategy = split_strategy 
        self.split_strategy = split_strategy 
        self.root=None
        
    def fit(self, x_vals, y_vals):
        """
        Aufruf um Decision Tree zu erstellen
        -> Parameter: Trainings-Datenset mit Features und Labels
        
        Features-Count = Anzahl an Spalten in Datenset
        Root=Oberster Knoten in Baum mit Verweis auf Kindsknoten
        """
        if not self.features_count:
            self.features_count = x_vals.shape[1]
        else:
            self.features_count = min(x_vals.shape[1], self.features_count)

        self.root = self.create_tree(x_vals, y_vals)
    
    def create_tree(self, x_vals, y_vals, depth=0):
        """
        Erstellen des Baums
        Parameter: - Trainigs-Datenset für Features und Labels
                   - Neue Variable für aktuelle Tiefe
        """
        #Anzahl Zeilen, Anzahl Spalten
        n_samples, n_feats = x_vals.shape
        #Klassen
        n_labels = len(np.unique(y_vals))
        #Abbruchkriterium
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self.most_frequent_label(y_vals)
            return CustomNode(value=leaf_value)
        
        features = np.random.choice(n_feats, self.features_count, replace=False)
        #Best Split ermitteln
        
        best_feature, best_treshold = self.choose_split(x_vals, 
                                                        y_vals, 
                                                        features, 
                                                        self.split_strategy)
        #Aufbauen des Baums
        id_left, id_right = self.split(x_vals[:, best_feature], best_treshold)
        left = self.create_tree(x_vals[id_left, :], y_vals[id_left], depth+1)
        right = self.create_tree(x_vals[id_right, :], y_vals[id_right], depth+1)
        return CustomNode(best_feature, best_treshold, left, right)
    
    def traverse_tree(self, x_vals, node):
        """
        Methode zum Durchlaufen des Decision Trees
        """
        if node.is_leave():
            return node.value
        
        if x_vals[node.feature_id] <= node.treshold:
            return self.traverse_tree(x_vals, node.left)
        return self.traverse_tree(x_vals, node.right)
    
    def predict(self, x_vals):
        """
        Vorhersagen für Datenset treffen, durch ablaufen des Trees
        """
        return np.array([self.traverse_tree(x_val, self.root)for x_val in x_vals])
    
    def choose_split(self, x_vals, y_vals, features, strategy):
        """
        Geeignetsten Datensplit basierend auf gewünschter Methode finden
        """
        best_gain = -1
        split_number, split_treshold = None, None 

        for feat in features:
            x_column = x_vals[:, feat]
            
            if strategy == SplitStrategy.MEDIAN:
                treshold = np.median(x_column)
            elif strategy == SplitStrategy.DURCHSCHNITT:
                treshold = np.mean(x_column)
            elif strategy == SplitStrategy.RANDOM_SPLIT:
                treshold = np.random.choice(x_column)
            else:  # Standardwert Best Split
                tresholds = np.unique(x_column)
                for treshold in tresholds:
                    gain = self.calc_information_gain(y_vals, x_column, treshold)
                    weighted_gain=self.feature_weights*gain
                    if weighted_gain > best_gain:
                        best_gain = weighted_gain
                        split_number = feat
                        split_treshold = treshold
                continue  # Rest der Schleife überspringen
            
            gain = self.calc_information_gain(y_vals, x_column, treshold)
            if gain > best_gain:
                best_gain = gain
                split_number = feat
                split_treshold = treshold

        return split_number, split_treshold 
    
    def calc_information_gain(self, y_vals, x_column, treshold):
        """
        Berechnung des Information Gain
        IG=E(parent)-[weighted averege]*E(children)
        """
        parent_entropy = self.calc_entropy(y_vals)
        id_left, id_right = self.split(x_column, treshold)
        if len(id_left) == 0 or len (id_right)==0:
            return 0
        #Gewichtete Entropy vom Kindknoten
        n_val = len(y_vals)
        n_l, n_r = len(id_left), len(id_right)
        e_l, e_r = self.calc_entropy(y_vals[id_left]), self.calc_entropy(y_vals[id_right])
        child_entropy = (n_l/n_val) * e_l + (n_r/n_val) * e_r
        
        #Information Gain berechnen
        information_gain = parent_entropy-child_entropy
        return information_gain
        
    def calc_entropy(self, y_vals):
        """
        Berechnen der Entropy
        """
        # -Sum of p(X)*log2(p(X)), P(X) = Number of x/number of values
        count_numbers = np.bincount(y_vals) #Array wie oft welcge Nummer verwendet
        p_vals = count_numbers/len(y_vals)
        return -np.sum([p*np.log2(p) for p in p_vals if p>0])  
    def split (self, x_column, split_treshold):
        """
        Split Tree
        """
        id_left = np.argwhere(x_column<=split_treshold).flatten()
        id_right = np.argwhere(x_column>split_treshold).flatten()
        return id_left, id_right
    
    def most_frequent_label(self, y_vals):
        """
        Meistverwendete Label zurückgeben
        """
        counter = Counter(y_vals)
        value = counter.most_common(1)[0][0]
        return value
    def root_node(self):
        """
        Zurückgeben des Root Nodes
        """
        return self.root
    
    def cnodes_2_node_structure(self, node=None):
        """
        CustomNode in Node für Rückgabemodell transformieren
        """
        if node is None:
            node = self.root
        if node.is_leave():
            return LeaveNode(value=node.value)
        left_child = self.cnodes_2_node_structure(node.left)
        right_child = self.cnodes_2_node_structure(node.right)
        return FeatureNode(feature_id=node.feature_id, 
                           treshold=node.treshold, 
                           left=left_child, 
                           right=right_child, 
                           feature_name=f"Is feature {node.feature_id} <= {node.treshold}?")

    def node_error(self, y_vals):
        """
        Berechnet den Fehler eines Knotens basierend auf den gegebenen Zielwerten.

        Parameters:
        - y: Die Zielwerte.

        Returns:
        - error: Der Fehler des Knotens.
        """
        
        most_common_label = self.most_frequent_label(y_vals)
        #error = sum([1 for label in y_vals if label != most_common_label])
        error = sum(1 for label in y_vals if label != most_common_label)
        return error

    def prune(self, node: CustomNode, x_vals, y_vals):
        """
        Stutzt den Baum rekursiv, um Overfitting zu verhindern.

        Parameters:
        - node: Der aktuelle Knoten.
        - X: Die Eingabedaten.
        - y: Die Zielwerte.
        """
        if node.is_leave():
            return 
        id_left, id_right = self.split(x_vals[:, node.feature_id], node.treshold)
        if node.left:
            self.prune(node.left, x_vals[id_left], y_vals[id_left])
        if node.right:
            self.prune(node.right, x_vals[id_right], y_vals[id_right])
        if node.left.is_leave() and node.right.is_leave():
            error_without_prune = self.node_error(y_vals)
            error_with_prune = self.node_error(y_vals[id_left]) + self.node_error(y_vals[id_right])     
            if error_with_prune < error_without_prune:
                node.left = None
                node.right = None
                node.value = self.most_frequent_label(y_vals) 
    
    
    