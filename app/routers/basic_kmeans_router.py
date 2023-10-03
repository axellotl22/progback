"""
basic_kmeans_router.py
----------------------
Router for performing KMeans clustering using the optimized KMeans and MiniBatch KMeans.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from app.models.basic_kmeans_model import KMeansInput, KMeansResult
from app.services.basic_kmeans_service import perform_kmeans_clustering
from app.services.utils import save_temp_file, delete_file

router = APIRouter()

@router.post("/perform-kmeans", response_model=KMeansResult)
async def perform_kmeans(
    input_data: KMeansInput,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        # Save the uploaded file temporarily
        file_path = save_temp_file(file, "tmp/")
        
        result = perform_kmeans_clustering(
            input_data=input_data, 
            filename=file_path, 
            user_id=input_data.user_id, 
            request_id=input_data.request_id,
            column1=input_data.column1, 
            column2=input_data.column2,
            method=input_data.method,
            distance_metric=input_data.distance_metric
        )

        # Schedule file deletion in the background
        background_tasks.add_task(delete_file, file_path)

        return result

    except Exception as e:
        # Handle exceptions gracefully and return an error response
        raise HTTPException(status_code=500, detail=str(e))
