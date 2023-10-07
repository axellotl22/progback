"""
basic_kmeans_router.py
----------------------
API router for performing KMeans clustering.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.basic_kmeans_service import perform_kmeans_from_file
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()


@router.post("/perform-2d-kmeans/")
# pylint: disable=too-many-arguments
# pylint: disable=R0801
async def kmeans(
    file: UploadFile = File(...),
    column1: int = Query(0,
                          description="Index of the first column"),
    column2: int = Query(1,
                          description="Index of the second column"),
    distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query("OptimizedKMeans",
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    k_clusters: int = Query(2, description="Number of clusters"),
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID")
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
    
    Returns:
    - KMeansResult: Result of the KMeans clustering.
    """
    try:
        kmeans_result = perform_kmeans_from_file(
            file=file,
            user_k=k_clusters,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            selected_columns=[column1, column2]
        )
        # Return the KMeansResult object.
        return kmeans_result
    # pylint: disable=duplicate-code
    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
