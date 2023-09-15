""" Router für Clustering-Endpunkte. """

import os
import logging

from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.clustering_model import ClusterResult, ClusterPoint
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering, delete_file
)

# Überprüfen, ob sich die App im Testmodus befindet
TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"

router = APIRouter()
TEMP_FILES_DIR = "temp_files/"

@router.post("/upload/", response_model=ClusterResult)
async def upload_file(file: UploadFile = File(...), clusters: Optional[int] = None):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    die Clustering-Ergebnisse zurück. Der Benutzer kann optional 
    die Anzahl der Cluster bestimmen.
    """
    # Überprüfen und Erstellen des temporären Dateiverzeichnisses bei Bedarf
    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    file_path = os.path.join(TEMP_FILES_DIR, file.filename)

    # Speichern der hochgeladenen Datei im TEMP_FILES_DIR
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        # Lade die Daten aus der hochgeladenen Datei in einen DataFrame
        data_frame = load_dataframe(file_path)
        
        # Bereinige den DataFrame (z.B. Entfernen von NaN-Werten)
        data_frame = clean_dataframe(data_frame)
        
        # Wenn der Benutzer keine Clusteranzahl bestimmt hat, wird die optimale Anzahl ermittelt
        if clusters is None:
            optimal_clusters = determine_optimal_clusters(data_frame)
        else:
            # Überprüfen auf ungültige Clusteranzahlen
            if clusters <= 1 or clusters > len(data_frame):
                raise HTTPException(400, "Ungültige Anzahl von Clustern")
            optimal_clusters = clusters
        
        # Durchführen des KMeans-Clusterings mit der angegebenen oder bestimmten Anzahl von Clustern
        kmeans = perform_clustering(data_frame, optimal_clusters)

        # Bestimme die Zentroide der Cluster
        centroids = [
            ClusterPoint(x=c[0], y=c[1], cluster=i)
            for i, c in enumerate(kmeans.cluster_centers_)
        ]

        # Bestimme die Datenpunkte und ihren zugehörigen Cluster
        points = [
            ClusterPoint(x=p[0], y=p[1], cluster=l)
            for p, l in zip(data_frame.values, kmeans.labels_)
        ]

        # Karte zur Zuordnung von Datenpunkten zu ihren Zentroiden
        point_to_centroid = dict(zip(range(len(points)), kmeans.labels_))
        
        return ClusterResult(
            points=points,
            centroids=centroids,
            point_to_centroid_mappings=point_to_centroid
        )

    # Fehlerbehandlung für verschiedene Szenarien
    except ValueError as error:
        logging.error("Fehler beim Lesen der Datei: %s", error)
        raise HTTPException(400, "Nicht unterstützter Dateityp") from error

    except Exception as error:
        logging.error("Fehler bei der Dateiverarbeitung: %s", error)
        raise HTTPException(500, "Fehler bei der Dateiverarbeitung") from error

    # Schließlich, lösche die temporäre Datei, wenn nicht im Testmodus
    finally:
        if not TEST_MODE:
            delete_file(file_path)
