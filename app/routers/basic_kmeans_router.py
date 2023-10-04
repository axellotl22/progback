"""
basic_kmeans_router.py
----------------------
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.basic_kmeans_service import perform_kmeans
from app.services.custom_kmeans import BaseOptimizedKMeans

router = APIRouter()

@router.post("/perform-kmeans/")
async def kmeans(
    file: UploadFile = File(...),
    column_1: int = Query(0, alias="Spalte 1", description="Index der ersten Spalte"),
    column_2: int = Query(1, alias="Spalte 2", description="Index der zweiten Spalte"),
    distance_metric: str = Query(..., 
                                 description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
    kmeans_type: str = Query(..., 
                             description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
    n_clusters: int = Query(..., description="Die Anzahl der Cluster"),
    user_id: int = Query(0, description="Benutzer-ID"),
    request_id: int = Query(0, description="Anfrage-ID")
):
    """Endpunkt für KMeans-Clustering."""
    try:
        kmeans_result = perform_kmeans(
            file,
            n_clusters,
            distance_metric,
            kmeans_type,
            user_id,
            request_id,
            selected_columns=[column_1, column_2]
        )
        # Rückgabe des KMeansResult-Objekts.
        return kmeans_result.dict()

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=400)
