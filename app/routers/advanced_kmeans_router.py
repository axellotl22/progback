"""
advanced_kmeans_router.py
-------------------------
API router for performing KMeans clustering with automatic k determination.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.advanced_kmeans_service import perform_advanced_kmeans
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-advanced-2d-kmeans/")
# pylint: disable=too-many-arguments

async def advanced_kmeans(
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
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID"),
    normalize: bool = True
):
    """
    Performs KMeans clustering on a 2D dataset and determines the optimal number of clusters (k) 
    automatically.

    Args:
    - file (UploadFile): The dataset file uploaded by the user.
    - column1 (int): Index of the first data column to be considered.
    - column2 (int): Index of the second data column to be considered.
    - distance_metric (str): The distance measure used during clustering. Choice impacts cluster 
                             formation and results.
    - kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional, 
                         while 'OptimizedMiniBatchKMeans' is faster but approximative.
    - user_id (int): Identifier for tracking user requests.
    - request_id (int): Identifier for individual API calls.
    - normalize (bool): Whether to normalize data before clustering. Default is True.

    Returns:
    - KMeansResult: A structured result of the clustering process, including cluster centers and 
                    member points.
    """
    try:
        kmeans_result = perform_advanced_kmeans(
            file,
            distance_metric,
            kmeans_type,
            user_id,
            request_id,
            selected_columns=[column1, column2],
            normalize= normalize
        )
        # Return the KMeansResult object.
        return kmeans_result

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
