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
    request_id: int = Query(0, description="Request ID")
):
    """
    Endpoint for KMeans clustering with automatic k determination.

    Args:
    - file (UploadFile): Uploaded data file.
    - column_1 (int): Index of the first column.
    - column_2 (int): Index of the second column.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - user_id (int): User ID.
    - request_id (int): Request ID.

    Returns:
    - KMeansResult: Result of the KMeans clustering.
    """
    try:
        kmeans_result = perform_advanced_kmeans(
            file,
            distance_metric,
            kmeans_type,
            user_id,
            request_id,
            selected_columns=[column1, column2]
        )
        # Return the KMeansResult object.
        return kmeans_result

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
