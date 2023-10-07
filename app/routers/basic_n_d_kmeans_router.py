"""
basic_nd_kmeans_router.py
-------------------
API router for performing N-Dimensional KMeans clustering and reducing to 2D.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.n_d_basic_kmeans_service import perform_nd_kmeans_from_file
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-nd-kmeans/")
# pylint: disable=too-many-arguments
# pylint: disable=duplicate-code
async def kmeans_nd(
    file: UploadFile = File(...),
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
    Endpoint for N-D KMeans clustering with dimensionality reduction to 2D.

    Args:
    - file (UploadFile): Uploaded data file.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - n_clusters (int): Number of clusters.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    
    Returns:
    - KMeansResultND: Result of the N-D KMeans clustering reduced to 2D.
    """
    try:
        kmeans_result_nd = perform_nd_kmeans_from_file(
            file=file,
            user_k=k_clusters,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id
        )
        # Return the KMeansResultND object.
        return kmeans_result_nd

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type or invalid parameters") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
