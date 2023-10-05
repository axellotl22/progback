"""
basic_kmeans_service.py
-----------------------
Service for performing KMeans clustering using optimized KMeans and MiniBatch KMeans.
"""

import logging
import os
from typing import Union
import pandas as pd
import numpy as np
from fastapi import UploadFile
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import BasicKMeansResult, Cluster, Centroid
from app.services.utils import (load_dataframe, clean_dataframe, save_temp_file, 
                                delete_file, extract_selected_columns)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_to_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the Cluster model structure.

    Args:
    - data_frame (pd.DataFrame): DataFrame containing the clustered data.
    - cluster_centers (np.ndarray): Array of cluster centers.

    Returns:
    - list: List of Cluster models.
    """
    clusters_list = []

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_points = data_frame[data_frame["cluster"] == cluster_id].drop(
            columns=["cluster"]).to_dict(orient="records")
        clusters_list.append(
            Cluster(
                cluster_nr=cluster_id,
                centroid=Centroid(
                    x=cluster_centers[cluster_id][0], y=cluster_centers[cluster_id][1]),
                points=cluster_points
            )
        )

    return clusters_list


# pylint: disable=too-many-arguments
# pylint: disable=R0914
def perform_kmeans(
    file: UploadFile,
    k: int,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None
) -> BasicKMeansResult:
    """
    Perform KMeans clustering on an uploaded file.

    Args:
    - file (UploadFile): Uploaded data file.
    - k (int): Number of clusters.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - selected_columns (list[int]): Indices of selected columns.

    Returns:
    - KMeansResult: Result of the KMeans clustering.
    """
    # Save and load the uploaded file
    temp_file_path = save_temp_file(file, "temp/")
    logger.info("Saved temp file to: %s", temp_file_path)
    
    data_frame = load_dataframe(temp_file_path)
    logger.info("Loaded data from temp file. Shape: %s", data_frame.shape)

    data_frame = clean_dataframe(data_frame)
    logger.info("Cleaned data. New shape: %s", data_frame.shape)

    # Select specific columns if provided
    if selected_columns:
        data_frame = extract_selected_columns(data_frame, selected_columns)
        logger.info("Selected columns. New shape: %s", data_frame.shape)

    # Convert DataFrame to numpy array for clustering
    data_np = data_frame.values
    logger.info("Converted data to numpy array. Shape: %s", data_np.shape)

    # Initialize the KMeans model
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(k, distance_metric)
    else:
        raise ValueError("Invalid kmeans_type: {}".format(kmeans_type))
    logger.info("Initialized %s model.", kmeans_type)

    # Fit the model
    model.fit(data_np)
    logger.info("Fitted the model.")

    # Add cluster assignments to the DataFrame using _assign_labels method
    data_frame['cluster'] = model.assign_labels(data_np)
    logger.info("Assigned labels to data.")

    # Transform the results to the Cluster model structure
    clusters = transform_to_cluster_model(data_frame, model.cluster_centers_)
    logger.info("Transformed data to Cluster models.")

    x_label = data_frame.columns[0]
    y_label = data_frame.columns[1]

    # Cleanup temp file
    delete_file(temp_file_path)
    logger.info("Deleted temp file: %s", temp_file_path)

    logger.info("Completed perform_kmeans function.")
    return BasicKMeansResult(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        filename=os.path.splitext(file.filename)[0],
        k_value=k
    )

