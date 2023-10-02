""" Router für Classification-Endpunkte. """

import os
import logging

from typing import Optional, List, Union
from sklearn.model_selection import train_test_split
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.classification_decision_tree_model import TreeNode, DecisionTreeData, DecisionTreeResult, SplitStrategy
import app.services.classification_decision_tree_service as dts
import app.models.classification_decision_tree_model
from app.services.utils import clean_dataframe, select_columns, delete_file, load_dataframe
import numpy as np
import pandas as pd


TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()

@router.post("/perform-decision_tree-classification/", response_model=DecisionTreeResult)
async def perform_decision_tree_classification(
    file: UploadFile = File(...),
    min_samples_split: Optional[int]=2,
    max_depth: Optional[int]=100,
    split_strategy: Optional[SplitStrategy]=SplitStrategy.BEST_SPLIT,
    features_count: Optional[int]=None,
    feature_weights: Optional[Union[str, List[int]]] = None,
    presorted: Optional [int]=0
):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    den Entscheidungsbaum zurück. Der Benutzer kann optional 
    die zu berücksichtigenden Spalten bestimmen.
    """
    
    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    file_path = os.path.join(TEMP_FILES_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        data_frame = load_dataframe(file_path)
        data_frame = clean_dataframe(data_frame)
        data_frame= dts.convert_text_to_categorical(data_frame)

        
        if min_samples_split is None:
            min_samples_split=2
        if max_depth is None:
            max_depth=100
        if split_strategy is None:
            split_strategy=SplitStrategy.BEST_SPLIT
        if feature_weights is None:
            feature_weights= np.ones(features_count)
        features_count = None
        
        target_column=("Drug")
        #target_column="Education"

        # Features und Labels extrahieren
        X = data_frame.drop(target_column, axis=1).values  # Target Column Name = Zu klassifizierende Spalte
        y = data_frame[target_column].values

        # Feature-Namen extrahieren
        #feature_names = df.columns[:-1].tolist()
        #data = datasets.load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        clf=DecisionTreeResult(min_samples_split=min_samples_split, max_depth=max_depth, split_strategy=split_strategy)
        dts.fit(clf, X_train, y_train)
        
        #clf = DecisionTree(split_strategy=SplitStrategy.DURCHSCHNITT)
        #clf.fit(X_train, y_train)
        return DecisionTreeResult(clf, min_samples_split, max_depth, features_count, None, feature_weights, split_strategy)
    
    
    except ValueError as error:
        logging.error("Fehler beim Lesen der Datei: %s", error)
        raise HTTPException(400, "Nicht unterstützter Dateityp") from error

    except Exception as error:
        logging.error("Fehler bei der Dateiverarbeitung: %s", error)
        raise HTTPException(500, "Fehler bei der Dateiverarbeitung") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
