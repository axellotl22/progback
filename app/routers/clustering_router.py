""" Router für Clustering-Endpunkte. """

import os
import logging

from typing import Optional, Union

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.clustering_model import ClusterResult
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, select_columns, determine_optimal_clusters,
    perform_clustering, delete_file
)

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()

@router.post("/perform-kmeans-clustering/", response_model=ClusterResult)
async def perform_kmeans_clustering(
    file: UploadFile = File(...),
    clusters: Optional[int] = None,
    column1: Optional[Union[str, int]] = None,
    column2: Optional[Union[str, int]] = None,
    distance_metric: Optional[str] = Query("EUCLIDEAN", alias="distanceMetric", description="Mögliche Distanzmaße: EUCLIDEAN, MANHATTAN, CHEBYSHEV, MINKOWSKI")
):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    die Clustering-Ergebnisse zurück. Der Benutzer kann optional 
    die Anzahl der Cluster, die zu berücksichtigenden Spalten 
    und das Distanzmaß bestimmen.
    """
    supported_distance_metrics = ["EUCLIDEAN", "MANHATTAN", "CHEBYSHEV", "MINKOWSKI"]
    
    # Überprüfen Sie, ob das bereitgestellte Distanzmaß unterstützt wird
    if distance_metric not in supported_distance_metrics:
        raise HTTPException(400, f"Ungültiges Distanzmaß. Unterstützte Maße sind: {', '.join(supported_distance_metrics)}")

    # Umwandeln von column1 und column2 in Integer, wenn sie Strings sind
    if isinstance(column1, str):
        column1 = int(column1)
    if isinstance(column2, str):
        column2 = int(column2)

    columns = [column1, column2] if column1 is not None and column2 is not None else None

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

        clustering_results = perform_clustering(data_frame, optimal_clusters, distance_metric)

        return ClusterResult(
            name=f"K-Means Ergebnis von: {os.path.splitext(file.filename)[0]}",
            cluster=clustering_results["cluster"],
            x_label=clustering_results["x_label"],
            y_label=clustering_results["y_label"],
            iterations=clustering_results["iterations"],
            distance_metric=distance_metric
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
