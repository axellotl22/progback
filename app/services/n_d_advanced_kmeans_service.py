"""
advanced_n_d_kmeans_service.py
------------------------------
Service for performing N-Dimensional KMeans clustering.
Automatic k determination using silhouette scores.
"""

from typing import Union

from sklearn.manifold import TSNE
import pandas as pd
from fastapi import UploadFile

from app.models.custom_kmeans_model import BasicKMeansResult, KMeansResult3D
from app.services.n_d_basic_kmeans_service import perform_nd_kmeans_from_dataframe
from app.services.utils import process_uploaded_file, normalize_dataframe, handle_categorical_data
from app.services.advanced_kmeans_service import determine_optimal_k

# pylint: disable=too-many-arguments,too-many-locals
def perform_advanced_nd_kmeans(
    file: UploadFile,
    distance_metric: str,
    kmeans_type: str,
    user_id: int,
    request_id: int,
    selected_columns: Union[None, list[int]] = None,
    use_3d_model: bool = False
) -> Union[BasicKMeansResult, KMeansResult3D]:
    """
    Perform N-Dimensional KMeans clustering on an uploaded file with automatic k determination.
    """
    # Process the uploaded file
    data_frame, filename = process_uploaded_file(file, selected_columns)
    
    # Convert categorical data
    data_frame_converted = handle_categorical_data(data_frame)
    
    # Normalize the dataframe
    data_frame_norm = normalize_dataframe(data_frame_converted)

    # Reduce to 2D or 3D using t-SNE based on the use_3d_model flag
    n_components = 3 if use_3d_model else 2
    tsne = TSNE(n_components=n_components, random_state=42)
    data_reduced = tsne.fit_transform(data_frame_norm)

    # Determine the optimal k in reduced space (2D or 3D)
    max_clusters = min(int(0.25 * data_reduced.shape[0]), 20)
    optimal_k = determine_optimal_k(pd.DataFrame(data_reduced), max_clusters)

    # Use the n_d_basic_kmeans_service with the determined optimal k
    result = perform_nd_kmeans_from_dataframe(
        data_frame=data_frame_converted,
        distance_metric=distance_metric,
        kmeans_type=kmeans_type,
        user_id=user_id,
        request_id=request_id,
        user_k=optimal_k,
        filename=filename,
        use_3d_model=use_3d_model
    )
    return result
