""" Router für Clustering-Endpunkte. """

import os
import logging

from typing import Optional, List, Union

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.clustering_model import ClusterResult, ClusterPoint
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, select_columns, determine_optimal_clusters,
    perform_clustering, delete_file, process_columns_input
)

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()

@router.post("/perform-kmeans-clustering/", response_model=ClusterResult)
async def perform_kmeans_clustering(
    file: UploadFile = File(...),
    clusters: Optional[int] = None,
    columns: Optional[Union[str, List[int]]] = None
):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    die Clustering-Ergebnisse zurück. Der Benutzer kann optional 
    die Anzahl der Cluster und die zu berücksichtigenden Spalten bestimmen.
    """
    if columns:
        columns = process_columns_input(columns)

    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    file_path = os.path.join(TEMP_FILES_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        data_frame = load_dataframe(file_path)
        data_frame = clean_dataframe(data_frame)

        if columns:
            data_frame = select_columns(data_frame, columns)

        if clusters is None:
            optimal_clusters = determine_optimal_clusters(data_frame)
        else:
            if clusters <= 1 or clusters > len(data_frame):
                raise HTTPException(400, "Ungültige Anzahl von Clustern")
            optimal_clusters = clusters

        clustering_results = perform_clustering(data_frame, optimal_clusters)

        centroids = [
            ClusterPoint(x=c["x"], y=c["y"], cluster=c["cluster"])
            for c in clustering_results["centroids"]
        ]
        points = [
            ClusterPoint(x=p["x"], y=p["y"], cluster=p["cluster"])
            for p in clustering_results["points"]
        ]

        return ClusterResult(
            x_label=clustering_results["x_label"],
            y_label=clustering_results["y_label"],
            points=points,
            centroids=centroids,
            point_to_centroid_mappings=clustering_results["point_to_centroid_mappings"]
        )

    except ValueError as error:
        logging.error("Fehler beim Lesen der Datei: %s", error)
        raise HTTPException(400, "Nicht unterstützter Dateityp") from error

    except Exception as error:
        logging.error("Fehler bei der Dateiverarbeitung: %s", error)
        raise HTTPException(500, "Fehler bei der Dateiverarbeitung") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
