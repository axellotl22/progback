"""Router for clustering endpoints."""

import os
import logging
from typing import Optional, Union, List
import numpy as np
from sklearn.model_selection import train_test_split

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.models.classification_model_decision_tree import DecisionTree, DecisionTreeResult, DecisionTreeTrainingsData, SplitStrategy, BestSplitStrategy
from app.services.classification_algorithms_decision_tree import CustomDecisionTree
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
    labelclassname: Optional[str]= Query(None, alias="ClassColumnName",description="Column4Classes"),
    feature_weights: Optional[List[int]]= Query(None, alias="FeatureWeights",description=""),
    presorted: Optional[int]= Query(None, alias="Vorsortieren?",description="Falls ja, Spaltennummer eintragen"),
    pruning: Optional[int]= Query(None, alias="Pruning?",description="YES, NO"),
    confusionMatrix: Optional[int]= Query(None, alias="ConfusionMatrix?",description="YES, NO"),
    test_size: Optional[float]= Query(None, alias="TestSize",description="Anteil an Testdaten"),
    random_state: Optional[int]= Query(None, alias="RandomState",description=""),
    best_split_strategy: Optional[BestSplitStrategy]= Query("Information Gain", alias="BestSplitStrategy",description="Information Gain, Entropy, Gini-Index"),
    feature_behaviour: Optional[int]= Query(None, alias="FeatureBehaviour",description="Mit/Ohne Zurücklegen -> Mehrfachwahl von Feature, YES/NO"),
    labelclassnumber: Optional[str]= Query(None, alias="ClassColumnNumber",description="Column4Classes")
):
    """
    This endpoint processes the uploaded file and returns
    the clustering results. User can optionally specify
    columns and distance metric.
    """

    # Validate distance metric
    supported_strats = list(SplitStrategy)
    if split_strategy not in supported_strats:
        error_msg = (
            f"Invalid distance metric. Supported metrics are: "
            f"{', '.join(supported_strats)}"
        )
        raise HTTPException(400, error_msg)
    supported_best_split_strats = list(BestSplitStrategy)
    if best_split_strategy not in supported_best_split_strats:
        error_msg = (
            f"Invalid distance metric. Supported metrics are: "
            f"{', '.join(supported_best_split_strats)}"
        )
        raise HTTPException(400, error_msg)
    # Convert columns to int if given as string
    if min_samples_split is None:
            min_samples_split=2
    if max_depth is None:
            max_depth=100
    if split_strategy is None:
            split_strategy=SplitStrategy.BEST_SPLIT
    #if feature_weights is None:
            #feature_weights= np.ones(features_count)
    feature_weights= np.ones(features_count)
    features_count = None
    # Process file
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    try:
        data_frame = dts.convert_text_to_categorical(load_dataframe(file_path))
        if presorted is not None:
            data_frame=dts.sort_data(data_frame, presorted)
        target_column=labelclassname
        X = data_frame.drop(target_column, axis=1).values  # Target Column Name = Zu klassifizierende Spalte
        y = data_frame[target_column].values
        if test_size is None:
            test_size=0.2
        if random_state is None:
            random_state=1234
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        clf = CustomDecisionTree(min_samples_split=min_samples_split, max_depth=max_depth, split_strategy=split_strategy)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        # Return clustering result model
        return DecisionTree(root=clf.CNodes2NodeStructure(), 
                            min_samples_split=clf.min_samples_split, 
                            max_depth=clf.max_depth, 
                            features_count=clf.features_count, 
                            labelclassname=labelclassname,
                            split_strategy=clf.split_strategy, 
                            accuracy=dts.accuracy(y_test, predictions))

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
