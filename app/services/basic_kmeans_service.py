"""
basic_kmeans_service.py
-----------------------
Service for performing KMeans clustering using the optimized KMeans and MiniBatch KMeans.
"""

import logging
import os
import pandas as pd
import numpy as np
from fastapi import UploadFile
from typing import Union

from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import KMeansResult, Cluster, Centroid
from app.services.utils import load_dataframe, clean_dataframe, select_columns, save_temp_file, delete_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def transform_to_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """Transform the data into the Cluster model structure."""
    clusters_list = []

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_points = data_frame[data_frame["cluster"] == cluster_id].drop(columns=["cluster"]).to_dict(orient="records")
        clusters_list.append(
            Cluster(
                clusterNr=cluster_id,
                centroid=Centroid(x=cluster_centers[cluster_id][0], y=cluster_centers[cluster_id][1]),
                points=cluster_points
            )
        )
    
    return clusters_list

def perform_kmeans(
    file: UploadFile, 
    k: int, 
    distance_metric: str, 
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None
) -> KMeansResult:

    # Save and load the uploaded file
    temp_file_path = save_temp_file(file, "temp/")
    data_frame = load_dataframe(temp_file_path)
    
    # Clean the dataframe
    data_frame = clean_dataframe(data_frame)
    
    # Select specific columns if provided
    if selected_columns:
        data_frame = select_columns(data_frame, selected_columns)
    
    # Convert DataFrame to numpy array for clustering
    data_np = data_frame.values
    
    # Initialize the KMeans model
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(n_clusters=k, distance_metric=distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(n_clusters=k, distance_metric=distance_metric)
    else:
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")
    
    # Fit the model
    model.fit(data_np)
    
    # Add cluster assignments to the DataFrame using _assign_labels method
    data_frame['cluster'] = model._assign_labels(data_np)
    
    # Transform the results to the Cluster model structure
    clusters = transform_to_cluster_model(data_frame, model.cluster_centers_)

    x_label = data_frame.columns[0] if selected_columns is None else data_frame.columns[selected_columns[0]]
    y_label = data_frame.columns[1] if selected_columns is None else data_frame.columns[selected_columns[1]]

    # Cleanup temp file
    delete_file(temp_file_path)
    
    return KMeansResult(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=model.max_iterations,
        used_distance_metric=distance_metric,
        filename=os.path.splitext(file.filename)[0],
        k_value=k
    )