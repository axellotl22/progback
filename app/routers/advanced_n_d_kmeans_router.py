"""
advanced_nd_kmeans_router.py
---------------------------
API router for performing advanced N-Dimensional KMeans clustering and determining optimal k automatically.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.n_d_advanced_kmeans_service import perform_advanced_nd_kmeans
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-advanced-nd-kmeans/")
# pylint: disable=too-many-arguments
# pylint: disable=duplicate-code
async def advanced_kmeans_nd(
    file: UploadFile = File(...),
    distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query("OptimizedKMeans",
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID")
):
    """
    Endpoint for advanced N-D KMeans clustering with dimensionality reduction to 2D and automatic k determination.

    Args:
    - file (UploadFile): Uploaded data file.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    
    Returns:
    - KMeansResultND: Result of the advanced N-D KMeans clustering reduced to 2D.
    """
    try:
        advanced_kmeans_result_nd = perform_advanced_nd_kmeans(
            file=file,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id
        )
        # Return the KMeansResultND object.
        return advanced_kmeans_result_nd

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type or invalid parameters") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
