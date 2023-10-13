"""
advanced_nd_kmeans_router.py
---------------------------
API router for performing advanced N-Dimensional KMeans clustering.
With determining optimal k automatically.
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
    request_id: int = Query(0, description="Request ID"),
    use_3d_model:bool = False
):
    """
    Performs advanced N-Dimensional KMeans clustering on the provided dataset. 
    Before clustering, t-SNE is employed for dimensionality reduction to 2D or 3D for visualization. 
    t-SNE is chosen due to its capability to preserve local structures in high-dimensional data, 
    making it ideal for visual representations. Furthermore, the optimal cluster count 'k' is 
    determined automatically.

    Args:
    - file (UploadFile): Dataset file to perform clustering on.
    - distance_metric (str): The metric used to measure distances during clustering.
    - kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional, 
                         while 'OptimizedMiniBatchKMeans' is faster but approximative.
    - user_id (int): Helps in tracking requests from specific users.
    - request_id (int): A unique label for individual API interactions.
    - use_3d_model (bool): If set to True, performs dimensionality reduction to 3D using t-SNE. 
                           Otherwise, reduces to 2D. Default is False.

    Returns:
    - KMeansResultND: Provides the clustering results reduced to 2D or 3D for visualization.
    """
    try:
        advanced_kmeans_result_nd = perform_advanced_nd_kmeans(
            file=file,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            use_3d_model=use_3d_model
        )
        # Return the KMeansResultND object.
        return advanced_kmeans_result_nd

    except ValueError as error:
        raise HTTPException(400, "Unsupported file type or invalid parameters") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
