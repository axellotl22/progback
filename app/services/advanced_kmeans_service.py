"""
advanced_kmeans_service.py
--------------------------
Service for performing KMeans clustering with automatic k determination using silhouette scores.
"""

from typing import Union

import numpy as np
from fastapi import UploadFile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.models.basic_kmeans_model import BasicKMeansResult
from app.services.basic_kmeans_service import perform_kmeans_from_dataframe
from app.services.utils import process_uploaded_file

# pylint: disable=duplicate-code
def determine_optimal_k(data_frame, max_clusters):
    """
    Determine the optimal number of clusters using silhouette score.
    """
    silhouette_scores = [silhouette_score(data_frame,
                                          KMeans(n_clusters=i,
                                                 init='k-means++',
                                                 max_iter=300,
                                                 n_init=10,
                                                 random_state=0).fit(data_frame).labels_)
                         for i in range(2, max_clusters+1)]

    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because we start calculating scores at k=2
    return optimal_k

# pylint: disable=too-many-arguments
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
    """    
    # Process the uploaded file
    data_frame, filename = process_uploaded_file(file, selected_columns)

    # Determine the optimal k
    max_clusters = min(int(0.25 * data_frame.shape[0]), 10)
    optimal_k = determine_optimal_k(data_frame, max_clusters)
    
    #print dataframe shape
    print(data_frame.shape)

    # Use the basic_kmeans_service with the determined optimal k
    result = perform_kmeans_from_dataframe(
        df=data_frame,
        distance_metric=distance_metric,
        kmeans_type=kmeans_type,
        user_id=user_id,
        request_id=request_id,
        advanced_k=optimal_k,
        filename=filename
    )
    return result
