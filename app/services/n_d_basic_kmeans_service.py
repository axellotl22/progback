"""
n_d_basic_kmeans_service.py
-----------------------
Service for performing N-Dimensional KMeans clustering using optimized KMeans and MiniBatch KMeans, 
reducing dimensionality to 2D using PCA.
"""

import logging
from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
from fastapi import UploadFile
from sklearn.decomposition import PCA
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import KMeansResultND, Cluster, Centroid
from app.services.utils import process_uploaded_file
from app.services.basic_kmeans_service import normalize_dataframe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def handle_categorical_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical and boolean columns to numerical format using one-hot encoding.
    """
    return pd.get_dummies(data_frame, drop_first=True)

def transform_to_2d_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the 2D Cluster model structure.
    """
    clusters_list = []
    for cluster_id in range(cluster_centers.shape[0]):
        cluster_data = data_frame[data_frame["cluster"] == cluster_id].drop(columns=["cluster"])
        cluster_points = [{"x": row.iloc[0], "y": row.iloc[1]} for _, row in cluster_data.iterrows()]
        
        clusters_list.append(
            Cluster(
                clusterNr=cluster_id,
                centroid=Centroid(
                    x=cluster_centers[cluster_id][0],
                    y=cluster_centers[cluster_id][1]),
                points=cluster_points
            )
        )
    return clusters_list

def extract_important_features(pca: PCA, n_features: int = 5) -> Dict[str, float]:
    """
    Extract the top n_features contributing to the PCA.
    """
    # Find the index of the important features based on explained_variance_ratio_
    important_indices = np.argsort(pca.explained_variance_ratio_)[::-1][:n_features]
    return {f"feature_{index}": importance for index, importance in zip(important_indices, pca.explained_variance_ratio_[important_indices])}

def perform_nd_kmeans_from_file(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    user_k: Optional[int] = None
) -> KMeansResultND:
    """
    Perform N-Dimensional KMeans clustering on an uploaded file and reduce to 2D using PCA.
    """
    data_frame, filename = process_uploaded_file(file, selected_columns)
    data_frame = handle_categorical_data(data_frame)
    return _perform_nd_kmeans(data_frame, filename, distance_metric, kmeans_type, user_id, request_id, user_k)

def perform_nd_kmeans_from_dataframe(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    user_k: Optional[int] = None
) -> KMeansResultND:
    """
    Perform N-Dimensional KMeans clustering on a DataFrame and reduce to 2D using PCA.
    """
    data_frame = handle_categorical_data(data_frame)
    return _perform_nd_kmeans(data_frame, filename, distance_metric, kmeans_type, user_id, request_id, user_k)

def _perform_nd_kmeans(
    data_frame: pd.DataFrame,
    filename: str,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    k: int
) -> KMeansResultND:
    logger.info(f"Starting _perform_nd_kmeans with filename={filename}, distance_metric={distance_metric}, kmeans_type={kmeans_type}, user_id={user_id}, request_id={request_id}, k={k}")
    
    original_columns = list(data_frame.columns)
    data_frame = normalize_dataframe(data_frame)
    data_np = data_frame.values
    
    logger.info(f"Data shape after normalization: {data_np.shape}")
    
    logger.info("Starting PCA...")
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_np)
    logger.info(f"Data shape after PCA: {data_2d.shape}")
    
    important_features = extract_important_features(pca)
    important_features_mapped = {original_columns[int(key.split("_")[1])]: value for key, value in important_features.items()}
    logger.info(f"Important features extracted: {important_features_mapped}")

    logger.info("Starting KMeans clustering...")
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(k, distance_metric)
    else:
        logger.error(f"Invalid kmeans_type provided: {kmeans_type}")
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")
    
    logger.info(data_2d[:10])
    
    model.fit(data_2d)
    data_frame['cluster'] = model.assign_labels(data_2d)
    logger.info(f"Finished KMeans clustering with {model.iterations_} iterations.")
    
    clusters = transform_to_2d_cluster_model(data_frame, model.cluster_centers_)
    
    return KMeansResultND(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label = "PCA1",
        y_label = "PCA2",
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        name=filename,
        k_value=k,
        important_features=important_features_mapped
    )