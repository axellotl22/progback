"""
three_d_basic_kmeans_service.py
-----------------------
Service for performing 3D KMeans clustering using optimized KMeans and MiniBatch KMeans.
"""

import logging
from typing import Optional, Union
import pandas as pd
from fastapi import UploadFile
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.custom_kmeans_model import KMeansResult3D
from app.services.utils import (process_uploaded_file, normalize_dataframe, 
                                handle_categorical_data, transform_to_3d_cluster_model)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=too-many-arguments
def perform_3d_kmeans_from_file(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    user_k: Optional[int] = None,
    normalize: bool = True
) -> KMeansResult3D:
    """
    Perform 3D KMeans clustering on an uploaded file.
    """
    data_frame, filename = process_uploaded_file(file, selected_columns)
    
    # categorical data
    data_frame=handle_categorical_data(data_frame)
    
    logger.info("Processed uploaded file. Shape: %s", data_frame.shape)
    return _perform_3d_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, user_k, normalize)

# pylint: disable=too-many-arguments


def perform_3d_kmeans_from_dataframe(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    advanced_k: Optional[int] = None,
    normalize: bool = True
) -> KMeansResult3D:
    """
    Perform 3D KMeans clustering on a DataFrame.
    """
    data_frame = handle_categorical_data(data_frame)

    return _perform_3d_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, advanced_k, normalize)

# pylint: disable=R0801
# pylint: disable=too-many-arguments
def _perform_3d_kmeans(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    k: int,
    normalize: bool = True
) -> KMeansResult3D:
    # Convert DataFrame to numpy array for clustering
    if normalize:
        data_frame = normalize_dataframe(data_frame)
    
    data_np = data_frame.values
    data_np = data_frame.values
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

    # Transform the results to the 3D Cluster model structure
    clusters = transform_to_3d_cluster_model(
        data_frame, model.cluster_centers_)
    logger.info("Transformed data to 3D Cluster models.")

    x_label = data_frame.columns[0]
    y_label = data_frame.columns[1]
    z_label = data_frame.columns[2]

    logger.info("Completed _perform_3d_kmeans function.")
    return KMeansResult3D(
        user_id=user_id,
        request_id=request_id,
        cluster=clusters,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        name=filename,
        k_value=k
    )
