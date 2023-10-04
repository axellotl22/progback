"""
basic_kmeans_router.py
----------------------
API router for performing KMeans clustering.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.basic_kmeans_service import perform_kmeans
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()


@router.post("/perform-2d-kmeans/")
# pylint: disable=too-many-arguments

async def kmeans(
    file: UploadFile = File(...),
    column_1: int = Query(0, alias="Column 1",
                          description="Index of the first column"),
    column_2: int = Query(1, alias="Column 2",
                          description="Index of the second column"),
    distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query("OptimizedKMeans",
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    n_clusters: int = Query(2, description="Number of clusters"),
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID"),
    auto_pca: bool = Query(
        True, description="Apply automatic PCA dimension reduction")
):
    """
    Endpoint for KMeans clustering.

    Args:
    - file (UploadFile): Uploaded data file.
    - column_1 (int): Index of the first column.
    - column_2 (int): Index of the second column.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - n_clusters (int): Number of clusters.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    - auto_pca (bool): Flag to enable automatic PCA.

    Returns:
    - KMeansResult: Result of the KMeans clustering.
    """
    try:
        kmeans_result = perform_kmeans(
            file,
            n_clusters,
            distance_metric,
            kmeans_type,
            user_id,
            request_id,
            selected_columns=[column_1, column_2],
            auto_pca=auto_pca
        )
        # Return the KMeansResult object.
        return kmeans_result
    # pylint: disable=duplicate-code
    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
