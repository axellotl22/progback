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
import numpy as np
from fastapi import UploadFile
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import BasicKMeansResult, Cluster, Centroid
from app.services.utils import process_uploaded_file


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_to_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the Cluster model structure.
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



def perform_kmeans_from_file(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    user_k: Optional[int] = None
) -> BasicKMeansResult:
    data_frame, filename = process_uploaded_file(file, selected_columns)
    logger.info("Processed uploaded file. Shape: %s", data_frame.shape)
    return _perform_kmeans(data_frame, filename, distance_metric, kmeans_type, user_id, request_id, user_k)


def perform_kmeans_from_dataframe(
    df: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    advanced_k: Optional[int] = None
) -> BasicKMeansResult:
    return _perform_kmeans(df, filename, distance_metric, kmeans_type, user_id, request_id, advanced_k)


def _perform_kmeans(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    k: int
) -> BasicKMeansResult:
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

    # Add cluster assignments to the DataFrame
    data_frame['cluster'] = model.assign_labels(data_np)
    logger.info("Assigned labels to data.")

    # Transform the results to the Cluster model structure
    clusters = transform_to_cluster_model(data_frame, model.cluster_centers_)
    logger.info("Transformed data to Cluster models.")

    x_label = data_frame.columns[0]
    y_label = data_frame.columns[1]

    logger.info("Completed perform_kmeans function.")
    return BasicKMeansResult(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        filename=filename,
        k_value=k
    )
