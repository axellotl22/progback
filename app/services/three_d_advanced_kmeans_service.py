"""
advanced_3d_kmeans_service.py
-----------------------------
Service for performing 3D KMeans clustering with automatic k determination using silhouette scores.
"""

from typing import Union
from fastapi import UploadFile

from app.models.basic_kmeans_model import KMeansResult3D
from app.services.three_d_basic_kmeans_service import perform_3d_kmeans_from_dataframe
from app.services.utils import process_uploaded_file, normalize_dataframe, handle_categorical_data
from app.services.advanced_kmeans_service import determine_optimal_k

# pylint: disable=too-many-arguments
def perform_advanced_3d_kmeans(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None
) -> KMeansResult3D:
    """
    Perform 3D KMeans clustering on an uploaded file with automatic k determination.
    """    
    # Process the uploaded file
    data_frame, filename = process_uploaded_file(file, selected_columns)

    data_frame_cat = handle_categorical_data(data_frame)
    
    data_frame_norm = normalize_dataframe(data_frame_cat)
    # Determine the optimal k
    max_clusters = min(int(0.25 * data_frame.shape[0]), 20)
    optimal_k = determine_optimal_k(data_frame_norm, max_clusters)

    # Use the three_d_basic_kmeans_service with the determined optimal k
    result = perform_3d_kmeans_from_dataframe(
        data_frame=data_frame,
        distance_metric=distance_metric,
        kmeans_type=kmeans_type,
        user_id=user_id,
        request_id=request_id,
        advanced_k=optimal_k,
        filename=filename
    )
    return result
