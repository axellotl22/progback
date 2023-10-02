"""
Dienste für Decision Tree Classification-Funktionen.
"""

#import ast
import logging
import os
from typing import List, Union

import numpy as np
import pandas as pd
from collections import Counter
from enum import Enum
from fastapi import HTTPException
from app.models.classification_decision_tree_model import DecisionTreeResult, SplitStrategy, TreeNode


# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10

def fit(self: DecisionTreeResult, X, y):
        self.features_count = X.shape[1] if not self.features_count else  min(X.shape[1], self.features_count)
        self.root = self.create_tree(X, y)

def create_tree(self: DecisionTreeResult, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        #Abbruchkriterium
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self.most_frequent_label(y)
            return TreeNode(value=leaf_value)
        
        features = np.random.choice(n_feats, self.features_count, replace=False)
        #Best Split ermitteln
        
        best_feature, best_treshold = self.choose_split(X, y, features, self.split_strategy)
        #Aufbauen des Baums
        id_left, id_right = self.split(X[:, best_feature], best_treshold)
        left = self.create_tree(X[id_left, :], y[id_left], depth+1)
        right = self.create_tree(X[id_right, :], y[id_right], depth+1)
        return TreeNode(best_feature, best_treshold, left, right)

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

            gain = self.calc_information_gain(self,y, X_column, treshold)
            if gain > best_gain:
                best_gain = gain
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
        
def convert_text_to_categorical(df: pd):
        """
        Convert all text columns in a DataFrame to categorical columns.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - DataFrame with text columns converted to categorical columns
        """
        for col in df.columns:
            if df[col].dtype == 'object':  # if column has text values
                df[col] = df[col].astype('category').cat.codes  # convert to categorical codes
        return df
            
def calc_entropy(self: DecisionTreeResult, y):
        # -Sum of p(X)*log2(p(X)), P(X) = Number of x/number of values
        count_numbers = np.bincount(y) #Array how often which number is used
        p_vals = count_numbers/len(y)
        return -np.sum([p*np.log2(p) for p in p_vals if p>0])    

def split (self: DecisionTreeResult, X_column, split_treshold):
        id_left = np.argwhere(X_column<=split_treshold).flatten()
        id_right = np.argwhere(X_column>split_treshold).flatten()
        return id_left, id_right
        
def calc_gini(self: DecisionTreeResult, y):
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
   
def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Lädt eine Datei in ein Pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    if file_path.endswith('.json'):
        return pd.read_json(file_path)
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file type: {os.path.splitext(file_path)[1]}")

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame von leeren und unvollständigen Zeilen.
    """
    return data_frame.dropna()
