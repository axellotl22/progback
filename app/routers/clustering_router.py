""" Router für Clustering-Endpunkte. """

import os
import logging
from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.clustering_model import ClusterResult
from app.services.clustering_service import process_and_cluster

from app.services.utils import (
    load_dataframe, delete_file, save_temp_file
)

from app.services.clustering_algorithms import CustomKMeans

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()


@router.post("/perform-kmeans-clustering/", response_model=ClusterResult)
async def perform_kmeans_clustering(
    file: UploadFile = File(...),
    clusters: Optional[int] = None,
    column1: Optional[Union[str, int]] = None,
    column2: Optional[Union[str, int]] = None,
    distance_metric: Optional[str] = Query(
        "EUCLIDEAN", alias="distanceMetric", 
        description=", ".join(CustomKMeans.SUPPORTED_DISTANCE_METRICS.keys())
)
):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    die Clustering-Ergebnisse zurück. Der Benutzer kann optional 
    die Anzahl der Cluster, die zu berücksichtigenden Spalten 
    und das Distanzmaß bestimmen.
    """
    supported_distance_metrics = list(CustomKMeans.SUPPORTED_DISTANCE_METRICS.keys())

    if distance_metric not in supported_distance_metrics:
        error_msg = (
            f"Ungültiges Distanzmaß. Unterstützte Maße sind: "
            f"{', '.join(supported_distance_metrics)}"
        )
        raise HTTPException(400, error_msg)

    if isinstance(column1, str):
        column1 = int(column1)
    if isinstance(column2, str):
        column2 = int(column2)

    columns = [
        column1, column2] if column1 is not None and column2 is not None else None
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    try:
        data_frame = load_dataframe(file_path)
        clustering_results = process_and_cluster(
            data_frame, clusters, distance_metric, columns)

        return ClusterResult(
        name=f"K-Means Ergebnis von: {os.path.splitext(file.filename)[0]}",
        cluster=clustering_results["cluster"],
        x_label=clustering_results["x_label"],
        y_label=clustering_results["y_label"],
        iterations=clustering_results["iterations"],
        distance_metric=distance_metric,
        silhouette_score=clustering_results["silhouette_score"],
        davies_bouldin_index=clustering_results["davies_bouldin_index"]
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
