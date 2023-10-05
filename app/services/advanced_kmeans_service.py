"""
advanced_kmeans_service.py
--------------------------
Service for performing KMeans clustering with automatic k determination using silhouette scores.
"""
import os
from fastapi import UploadFile
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.basic_kmeans_model import BasicKMeansResult
from app.services.utils import load_dataframe, clean_dataframe, extract_selected_columns, save_temp_file, delete_file
from app.services.basic_kmeans_service import transform_to_cluster_model


def silhouette_for_k(k, data, kmeans_type, distance_metric):
    """
    Compute the silhouette score for a given k.

    Args:
    - k (int): Number of clusters.
    - data (array-like): Data for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - distance_metric (str): Distance metric for clustering.

    Returns:
    - float: Silhouette score.
    """
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(k, distance_metric)
    else:
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")

    model.fit(data)
    labels = model.assign_labels(data)
    return silhouette_score(data, labels)


def determine_optimal_k(data, max_k, distance_metric="euclidean", kmeans_type="OptimizedKMeans"):
    """
    Determine the optimal number of clusters using the silhouette score.

    Args:
    - data (array-like): Data for clustering.
    - max_k (int): Maximum number of clusters to consider.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.

    Returns:
    - int: Optimal number of clusters.
    """
    results = Parallel(n_jobs=-1)(delayed(silhouette_for_k)(k, data, kmeans_type, distance_metric) for k in range(2, max_k + 1))
    optimal_k = results.index(max(results)) + 2  # +2 because k starts from 2
    return optimal_k


def perform_advanced_kmeans(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None
) -> BasicKMeansResult:
    """
    Perform KMeans clustering on an uploaded file with automatic k determination.

    Args:
    - file (UploadFile): Uploaded data file.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - selected_columns (list[int]): Indices of selected columns.
    - auto_pca (bool): Flag to enable automatic PCA.

    Returns:
    - KMeansResult: Result of the KMeans clustering.
    """

    # Save and load the uploaded file
    temp_file_path = save_temp_file(file, "temp/")
    data_frame = load_dataframe(temp_file_path)

    data_frame = clean_dataframe(data_frame)

    # Select specific columns if provided
    if selected_columns:
        data_frame = extract_selected_columns(data_frame, selected_columns)

    # Convert DataFrame to numpy array for clustering
    data_np = data_frame.values

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_np)

    # Determine the optimal k
    max_clusters = min(int(0.25 * data_frame.shape[0]), 50)
    optimal_k = determine_optimal_k(standardized_data, max_clusters, distance_metric, kmeans_type)

    # Initialize the KMeans model
    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(optimal_k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(optimal_k, distance_metric)
    else:
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")

    # Fit the model
    model.fit(standardized_data)

    # Add cluster assignments to the DataFrame using _assign_labels method
    data_frame['cluster'] = model.assign_labels(standardized_data)

    # Transform the results to the Cluster model structure
    clusters = transform_to_cluster_model(data_frame, model.cluster_centers_)

    x_label = data_frame.columns[0]
    y_label = data_frame.columns[1]

    # Cleanup temp file
    delete_file(temp_file_path)

    return BasicKMeansResult(
        user_id=user_id,
        request_id=request_id,
        clusters=clusters,
        x_label=x_label,
        y_label=y_label,
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        filename=os.path.splitext(file.filename)[0],
        k_value=optimal_k
    )

