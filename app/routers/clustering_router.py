"""
Router für Clustering-Endpunkte.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.clustering_model import ClusterResult
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering, delete_file
)
import logging

router = APIRouter()

@router.post("/upload/", response_model=ClusterResult)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"temp_files/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        df = load_dataframe(file_path)
        df = clean_dataframe(df)
        optimal_clusters = determine_optimal_clusters(df)
        labels = perform_clustering(df, optimal_clusters)

        delete_file(file_path)

        return {"cluster_labels": labels, "optimal_cluster_count": optimal_clusters}

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Error processing file.")