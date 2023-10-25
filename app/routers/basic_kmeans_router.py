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
    request_id: int = Query(0, description="Request ID"),
    normalize: bool = True
):
    """
    Endpoint for performing basic 2D KMeans clustering. The user can specify the desired 
    number of clusters and choose columns from the uploaded dataset for clustering. 

    Args:
    - file (UploadFile): Dataset uploaded by the user for clustering.
    - column_1, column_2 (int): Indices of columns to be used for 2D clustering.
    - distance_metric (str): Selected metric for measuring distances between data points.
    - kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional, 
                         while 'OptimizedMiniBatchKMeans' is faster but approximative.
    - n_clusters (int): Desired number of clusters.
    - user_id (int): ID associated with the user making the request.
    - request_id (int): ID specific to this clustering request.
    - normalize (bool): Whether to normalize data before clustering. Default is True.

    Returns:
    - KMeansResult: Object containing details of the KMeans clustering outcome.
    """
    try:
        kmeans_result = perform_kmeans_from_file(
            file=file,
            user_k=k_clusters,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            selected_columns=[column1, column2],
            normalize=normalize
        )
        asldklkasdfkjfdsajkhasdfjkhafsdjkafsd
        # Return the KMeansResult object.
        return kmeans_result
    # pylint: disable=duplicate-code
    except ValueError as error:
        raise HTTPException(400, f"Unsupported file type: {error}") from error

    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
