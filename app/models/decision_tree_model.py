""" Modelle für Classification. """
import numpy as np
from collections import Counter
class Node:
    def __init__(self, feature=None, treshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.treshold = treshold
        self.left = left
        self.right= right
        self.value= value
        
    def is_leave(self):
        return self.value is not None
        
        
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, features_count=None):
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.features_count = features_count
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
    
    
    def best_split (self, X, y, features):
        best_gain = -1
        split_number, split_treshold = None, None 
        
        for feat in features:
            X_column = X[:, feat]
            tresholds = np.unique (X_column)
            
            for treshold in tresholds:
                information_gain=self.calc_information_gain(y, X_column, treshold)
                
                if (information_gain > best_gain):
                    best_gain = information_gain
                    split_number = feat
                    split_treshold = treshold
                    
        return split_number, split_treshold
    
        
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
        
    #def calc_gini():
        
    #def calc_gini_index():   
    def most_frequent_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def traverse_tree(self, x, node):
        if node.is_leave():
            return node.value
        
        if x[node.feature] <= node.treshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root)for x in X])
    