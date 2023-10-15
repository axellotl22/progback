"""
three_d_kmeans_router.py
------------------------
API router for performing 3D KMeans clustering.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.three_d_basic_kmeans_service import perform_3d_kmeans_from_file
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-3d-kmeans/")
# pylint: disable=too-many-arguments
async def kmeans_3d(
    file: UploadFile = File(...),
    column1: int = Query(0,
                          description="Index of the first column"),
    column2: int = Query(1,
                          description="Index of the second column"),
    column3: int = Query(2,
                          description="Index of the third column"),
    distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query("OptimizedKMeans",
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    k_clusters: int = Query(2, description="Number of clusters"),
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID"),
    normalize: bool = Query(True, description="Normalize the data before clustering")
):
    """
    Basic 3D KMeans clustering endpoint. This endpoint allows the user to perform 
    KMeans clustering on three selected columns of the uploaded data file. 

    Args:
    - file (UploadFile): Data file uploaded by the user.
    - column_1, column_2, column_3 (int): Indices of the columns to be used for 3D clustering.
    - distance_metric (str): Chosen metric for measuring distances between data points.
    - kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional, 
                         while 'OptimizedMiniBatchKMeans' is faster but approximative.
    - k_clusters (int): Number of clusters to be formed.
    - user_id (int): ID associated with the user.
    - request_id (int): ID associated with this particular clustering request.
    - normalize (bool): Whether to normalize data before clustering. Default is True.

    Returns:
    - KMeansResult3D: Object detailing the result of the 3D KMeans clustering process.
    """
    try:
        kmeans_result_3d = perform_3d_kmeans_from_file(
            file=file,
            user_k=k_clusters,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            selected_columns=[column1, column2, column3],
            normalize=normalize
        )
        # Return the KMeansResult3D object.
        return kmeans_result_3d
    # pylint: disable=duplicate-code
    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
