"""
advanced_three_d_kmeans_router.py
---------------------------------
API router for performing advanced 3D KMeans clustering with automatic k determination.
"""


from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.three_d_advanced_kmeans_service import perform_advanced_3d_kmeans
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-advanced-3d-kmeans/")
# pylint: disable=too-many-arguments
# pylint: disable=duplicate-code
async def advanced_kmeans_3d(
    file: UploadFile = File(...),
    column1: int = Query(0, alias="Column 1",
                          description="Index of the first column"),
    column2: int = Query(1, alias="Column 2",
                          description="Index of the second column"),
    column3: int = Query(2, alias="Column 3",
                          description="Index of the third column"),
    distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query("OptimizedKMeans",
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    user_id: int = Query(0, description="User ID"),
    request_id: int = Query(0, description="Request ID")
):
    """
    Endpoint for advanced 3D KMeans clustering with automatic k determination.

    Args:
    - file (UploadFile): Uploaded data file.
    - column_1 (int): Index of the first column.
    - column_2 (int): Index of the second column.
    - column_3 (int): Index of the third column.
    - distance_metric (str): Distance metric for clustering.
    - kmeans_type (str): Type of KMeans model to use.
    - user_id (int): User ID.
    - request_id (int): Request ID.
    
    Returns:
    - KMeansResult3D: Result of the 3D KMeans clustering.
    """
    try:
        kmeans_result_3d = perform_advanced_3d_kmeans(
            file=file,
            distance_metric=distance_metric,
            kmeans_type=kmeans_type,
            user_id=user_id,
            request_id=request_id,
            selected_columns=[column1, column2, column3]
        )
        # Return the KMeansResult3D object.
        return kmeans_result_3d
    # pylint: disable=duplicate-code
    except ValueError as error:
        raise HTTPException(400, "Unsupported file type") from error
    except Exception as error:
        raise HTTPException(500, "Error processing file") from error
