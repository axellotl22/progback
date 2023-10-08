"""
three_d_basic_kmeans_service.py
-----------------------
Service for performing 3D KMeans clustering using optimized KMeans and MiniBatch KMeans.
"""

import logging
from typing import Optional, Union
import pandas as pd
import numpy as np
from fastapi import UploadFile
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import KMeansResult3D, Cluster3D, Centroid3D
from app.services.utils import process_uploaded_file, normalize_dataframe, handle_categorical_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_to_3d_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the 3D Cluster model structure.
    """
    clusters_list = []

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_data = data_frame[data_frame["cluster"] == cluster_id].drop(columns=[
                                                                            "cluster"])

        # Transform points to always have "x", "y", and "z" as keys
        cluster_points = [{"x": row[0], "y": row[1], "z": row[2]}
                          for _, row in cluster_data.iterrows()]

        clusters_list.append(
            Cluster3D(
                clusterNr=cluster_id,
                centroid=Centroid3D(
                    x=cluster_centers[cluster_id][0],
                    y=cluster_centers[cluster_id][1],
                    z=cluster_centers[cluster_id][2]),
                points=cluster_points
            )
        )

    return clusters_list


# pylint: disable=too-many-arguments
def perform_3d_kmeans_from_file(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    user_k: Optional[int] = None
) -> KMeansResult3D:
    """
    Perform 3D KMeans clustering on an uploaded file.
    """
    data_frame, filename = process_uploaded_file(file, selected_columns)
    
    # categorical data
    data_frame=handle_categorical_data(data_frame)
    
    logger.info("Processed uploaded file. Shape: %s", data_frame.shape)
    return _perform_3d_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, user_k)

# pylint: disable=too-many-arguments


def perform_3d_kmeans_from_dataframe(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    advanced_k: Optional[int] = None
) -> KMeansResult3D:
    """
    Perform 3D KMeans clustering on a DataFrame.
    """
    return _perform_3d_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, advanced_k)

# pylint: disable=R0801
# pylint: disable=too-many-arguments
def _perform_3d_kmeans(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    k: int
) -> KMeansResult3D:
    # Convert DataFrame to numpy array for clustering
    data_frame = normalize_dataframe(data_frame)
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
