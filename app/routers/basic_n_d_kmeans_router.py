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
    request_id: int = Query(0, description="Request ID"),
    use_3d_model:bool=False
):
    """
    Endpoint for performing KMeans clustering on N-Dimensional data. 
    Before clustering, the data is reduced in dimensionality 
    using t-SNE, which helps in visualizing multi-dimensional data in 2D or 3D space. 
    t-SNE is specifically used due to its 
    ability to maintain local structures and reveal patterns or clusters in the data.

    Args:
    - file (UploadFile): Dataset uploaded by the user for clustering.
    - distance_metric (str): Selected metric for measuring distances between data points.
    - kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional, 
                         while 'OptimizedMiniBatchKMeans' is faster but approximative.
    - n_clusters (int): Desired number of clusters.
    - user_id (int): ID associated with the user making the request.
    - request_id (int): ID specific to this clustering request.
    - use_3d_model (bool): If set to True, performs dimensionality reduction to 3D using t-SNE. 
                           Otherwise, reduces to 2D. Default is False.

    Returns:
    - KMeansResultND: Provides the clustering results reduced to 2D or 3D for visualization.
    """
    try:
        kmeans_result_nd = perform_nd_kmeans_from_file(
            file=file,
            user_k=k_clusters,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            use_3d_model=use_3d_model
        )
        # Return the KMeansResultND object.
        return kmeans_result_nd

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type or invalid parameters") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
