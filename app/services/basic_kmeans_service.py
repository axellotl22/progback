"""
basic_kmeans_service.py
-----------------------
Service for performing KMeans clustering using optimized KMeans and MiniBatch KMeans.
"""

# pylint: disable=too-many-arguments
# pylint: disable=R0914

import logging
from typing import Optional, Union
import pandas as pd
from fastapi import UploadFile

from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.custom_kmeans_model import BasicKMeansResult
from app.services.utils import (process_uploaded_file,normalize_dataframe, 
                                handle_categorical_data, transform_to_2d_cluster_model)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def perform_kmeans_from_file(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    user_k: Optional[int] = None,
    normalize: bool = True
) -> BasicKMeansResult:
    """
    Perform KMeans clustering on an uploaded file.
    """
    data_frame, filename = process_uploaded_file(file, selected_columns)
    logger.info("Processed uploaded file. Shape: %s", data_frame.shape)
    return _perform_kmeans(data_frame, filename, distance_metric, 
                           kmeans_type, user_id, request_id, user_k, normalize)


def perform_kmeans_from_dataframe(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    advanced_k: Optional[int] = None,
    normalize: bool = True
) -> BasicKMeansResult:
    """
    Perform KMeans clustering on a DataFrame.
    """
    return _perform_kmeans(data_frame, filename, distance_metric, 
                           kmeans_type, user_id, request_id, advanced_k, normalize)

# pylint: disable=R0801
def _perform_kmeans(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    k: int,
    normalize: bool = True
) -> BasicKMeansResult:
    data_frame_cat=handle_categorical_data(data_frame)
    
    if normalize:
        data_frame = normalize_dataframe(data_frame_cat)
    else: 
        data_frame = data_frame_cat
    
    data_np = data_frame.values
    
    logger.info(data_np[:5])
    
    logger.info("Converted data to numpy array. Shape: %s", data_np.shape)

    # Initialize the KMeans model
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(k, distance_metric)
    else:
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")

    logger.info("Initialized %s model.", kmeans_type)

    # Fit the model
    model.fit(data_np)
    logger.info("Fitted the model.")

    # Add cluster assignments to the DataFrame
    data_frame['cluster'] = model.assign_labels(data_np)
    logger.info("Assigned labels to data.")

    # Transform the results to the Cluster model structure
    clusters = transform_to_2d_cluster_model(data_frame, model.cluster_centers_)
    logger.info("Transformed data to Cluster models.")

    x_label = data_frame.columns[0]
    y_label = data_frame.columns[1]

    logger.info("Completed perform_kmeans function.")
    return BasicKMeansResult(
        user_id=user_id,
        request_id=request_id,
        cluster=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        name=filename,
        k_value=k
    )
