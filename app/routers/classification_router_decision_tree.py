"""Router for clustering endpoints."""

import os
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.models.classification_model_decision_tree import DecisionTree
from app.models.classification_model_decision_tree import SplitStrategy, BestSplitStrategy
from app.services.classification_algorithms_decision_tree import CustomDecisionTree
import app.services.classification_service_decision_tree as dts

from app.services.utils import (
    load_dataframe, delete_file, save_temp_file, handle_errors
)

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"
router = APIRouter()

@router.post("/perform-classification-decision-tree/", response_model=DecisionTree)

# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches
async def perform_classification_decision_tree(
    
    file: UploadFile = File(...),
    to_pred: UploadFile = File(default=None),
    min_samples_split: Optional[int] = Query(2, 
                alias="SampleCount4Split",  
                description="Anzahl an Dateneinträgen, um weiteren Split durchzuführen"),
    max_depth: Optional[int]= Query(100, alias="",description=""),
    split_strategy: Optional[SplitStrategy]= Query("Best Split", 
                                        alias="SplitStrategy",
                                    description="Best Split, Median, Durchschnitt, Random Split"),
    features_count: Optional[int]= Query(None, 
                                        alias="featureAmount",
                                        description=""),
    labelclassname: Optional[str]= Query(None, 
                                        alias="ClassColumnName",
                                        description="Column4Classes"),
   feature_weights: Optional[str]= Query(None, 
                                        alias="FeatureWeights",
                                        description=""),
    presorted: Optional[int]= Query(None, 
                                        alias="Vorsortieren?",
                                        description="Falls ja, Spaltennummer eintragen"),
    confusion_matrix: Optional[bool]= Query(None, 
                                        alias="ConfusionMatrix?",
                                        description="YES, NO"),
    test_size: Optional[float]= Query(None, 
                                        alias="TestSize",
                                        description="Anteil an Testdaten"),
    random_state: Optional[int]= Query(None, 
                                        alias="RandomState",
                                        description=""),
    best_split_strategy: Optional[BestSplitStrategy]= Query("Information Gain", 
                                        alias="BestSplitStrategy",
                                        description="Information Gain, Entropy, Gini-Index"),
    feature_behaviour: Optional[bool]= Query(None, 
                                        alias="FeatureBehaviour",
                                        description="Mehrfachwahl von Features erlauben"),
    labelclassnumber: Optional[int]= Query(None, 
                                        alias="ClassColumnNumber",
                                        description="Nummer der zu klassifizierenden Spalte"),
    pruning: Optional[bool]= Query(False, 
                                        alias="Pruning?",
                                        description="YES, NO")
):
    """
    This endpoint processes the uploaded file and returns
    the classification results. 
    """
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
    if min_samples_split is None:
        min_samples_split=2
    if max_depth is None:
        max_depth=100
    if split_strategy is None:
        split_strategy=SplitStrategy.BEST_SPLIT
    if split_strategy != SplitStrategy.BEST_SPLIT:
        best_split_strategy=BestSplitStrategy.NONE
    if feature_behaviour is None:
        feature_behaviour=False
    if pruning is None:
        pruning=False
    # Process file
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    try:
        data_frame = dts.convert_text_to_categorical(load_dataframe(file_path))
        if presorted is not None:
            data_frame = dts.sort_data(data_frame, presorted)

        if labelclassname is None:
            labelclassname = data_frame.columns[labelclassnumber]      
        target_column = labelclassname
        x_vals = data_frame.drop(target_column, axis=1).values
        y_vals = data_frame[target_column].values
        feature_names_df= data_frame.drop(target_column, axis=1)
        feature_names=feature_names_df.columns.tolist()
        if feature_weights is not None:
            #feature_weights = np.array([feature_weights.split(',')])
            feature_weights = [int(x) for x in feature_weights.split(",")]
        else:
            feature_weights = np.ones(x_vals.shape[1])
        if test_size is None:
            test_size=0.2
        if random_state is None:
            random_state=42
        x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=test_size, 
                                                            random_state=random_state)
        clf = CustomDecisionTree(min_samples_split=min_samples_split, 
                                 max_depth=max_depth, 
                                 split_strategy=split_strategy, 
                                 features_count=features_count,
                                 feature_behaviour=feature_behaviour,
                                 feature_weights=feature_weights)
        clf.fit(x_train, y_train)
        if pruning:
            clf.prune(x_vals=x_vals, y_vals=y_vals)
        
        predictions = clf.predict(x_test)
        
        if to_pred is not None:
            file_path_pred = save_temp_file(to_pred, TEMP_FILES_DIR)
            data_frame_pred = dts.convert_text_to_categorical(load_dataframe(file_path_pred))
            pred_array= data_frame_pred.values
            self_predictions = [str(pred) for pred in clf.predict(pred_array).tolist()]
        else: self_predictions=None
        if confusion_matrix:
            matrix=clf.confusion_matrix(y_true=y_test, y_pred=predictions)
        else:
            matrix=None

        return DecisionTree(root=clf.cnodes_2_node_structure(feature_names=feature_names), 
                            min_samples_split=clf.min_samples_split, 
                            max_depth=clf.max_depth, 
                            features_count=clf.features_count, 
                            labelclassname=labelclassname,
                            split_strategy=clf.split_strategy, 
                            accuracy=dts.accuracy(y_test, predictions),
                            feature_behaviour=feature_behaviour,
                            pruning=pruning,
                            confusion_matrix=matrix,
                            self_predictions=self_predictions,
                            feature_weights=feature_weights,
                            best_split_strategy=best_split_strategy,
                            random_state=random_state,
                            test_size=test_size,
                            feature_names=feature_names
                            )
    # pylint: disable=duplicate-code, broad-exception-caught
    except (ValueError, Exception) as error:
        handle_errors(error)
    finally:
        if not TEST_MODE:
            delete_file(file_path)
