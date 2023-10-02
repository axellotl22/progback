"""Router for clustering endpoints."""

import os
import logging
from typing import Optional, Union, List
import numpy as np
from sklearn.model_selection import train_test_split

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.models.classification_model_decision_tree import DecisionTree, DecisionTreeResult, DecisionTreeTrainingsData, SplitStrategy, BestSplitStrategy
from app.services.classification_algorithms_decision_tree import CustomDecisionTree, CustomNode
import app.services.classification_service_decision_tree as dts

from app.services.utils import (
    load_dataframe, delete_file, save_temp_file
)
from app.services.clustering_algorithms import CustomKMeans

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()


@router.post("/perform-classification-decision-tree/", response_model=DecisionTree)
# pylint: disable=too-many-arguments
async def perform_kmeans_clustering(
    file: UploadFile = File(...),
    min_samples_split: Optional[int] = Query(2, alias="SampleCount4Split",description="Anzahl an Dateneinträgen, um weiteren Split durchzuführen"),
    max_depth: Optional[int]= Query(100, alias="",description=""),
    split_strategy: Optional[SplitStrategy]= Query("Best Split", alias="SplitStrategy",description="Best Split, Median, Durchschnitt, Random Split"),
    features_count: Optional[int]= Query(None, alias="featureAmount",description=""),
    labelclass: Optional[int]= Query(None, alias="ClassColumn",description="Column4Classes"),
    feature_weights: Optional[List[int]]= Query(None, alias="FeatureWeights",description=""),
    presorted: Optional[int]= Query(None, alias="FeatureWeights",description="YES, NO"),
    pruning: Optional[int]= Query(None, alias="FeatureWeights",description="YES, NO")
    

):
    """
    This endpoint processes the uploaded file and returns
    the clustering results. User can optionally specify
    columns and distance metric.
    """

    # Validate distance metric
    supported_metrics = list(SplitStrategy)
    if split_strategy not in supported_metrics:
        error_msg = (
            f"Invalid distance metric. Supported metrics are: "
            f"{', '.join(supported_metrics)}"
        )
        raise HTTPException(400, error_msg)

    # Convert columns to int if given as string
    if min_samples_split is None:
            min_samples_split=2
    if max_depth is None:
            max_depth=100
    if split_strategy is None:
            split_strategy=SplitStrategy.BEST_SPLIT
    if feature_weights is None:
            feature_weights= np.ones(features_count)
    features_count = None
    # Process file
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    try:
        data_frame = dts.convert_text_to_categorical(load_dataframe(file_path))
        target_column=("Drug")
        X = data_frame.drop(target_column, axis=1).values  # Target Column Name = Zu klassifizierende Spalte
        y = data_frame[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        clf = DecisionTree()
        clf= dts.fit(clf,X_train, y_train)
        predictions = clf.predict(X_test)
        
        # Return clustering result model
        return DecisionTree(
            user_id=0,
            request_id=0,
            name=f"K-Means Result for: {os.path.splitext(file.filename)[0]}",
            root=None,
            min_samples_split= min_samples_split,
            max_depth= max_depth,
            features_count= features_count,
            labelclass= labelclass,
            feature_weights= feature_weights,
            split_strategy= split_strategy
        )
            
        

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
