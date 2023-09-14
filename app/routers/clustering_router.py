"""
Router für Clustering-Endpunkte.
"""

import logging
import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.clustering_model import ClusterResult
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering, delete_file
)

router = APIRouter()

TEMP_FILES_DIR = "temp_files"

@router.post("/upload/", response_model=ClusterResult)
async def upload_file(file: UploadFile = File(...)):
    """
    Lädt eine Datei hoch, führt Clustering darauf aus und gibt das Ergebnis zurück.
    """
    try:
        file_path = os.path.join(TEMP_FILES_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        data_frame = load_dataframe(file_path)
        data_frame = clean_dataframe(data_frame)
        optimal_clusters = determine_optimal_clusters(data_frame)
        labels = perform_clustering(data_frame, optimal_clusters)

        delete_file(file_path)

        return {"cluster_labels": labels, "optimal_cluster_count": optimal_clusters}

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(status_code=500, detail="Error processing file.") from error
