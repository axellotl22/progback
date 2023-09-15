""" Router für Clustering-Endpunkte. """

import os
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.clustering_model import ClusterResult, ClusterPoint
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering, delete_file
)

# Test Mode damit die Datei nicht gelöscht wird
TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"

router = APIRouter()
TEMP_FILES_DIR = "temp_files/"


@router.post("/upload/", response_model=ClusterResult)
async def upload_file(file: UploadFile = File(...)):
    """Verarbeitet die hochgeladene Datei und gibt die Clustering-Ergebnisse zurück."""
    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    file_path = os.path.join(TEMP_FILES_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        data_frame = load_dataframe(file_path)
        data_frame = clean_dataframe(data_frame)
        optimal_clusters = determine_optimal_clusters(data_frame)
        kmeans = perform_clustering(data_frame, optimal_clusters)

        centroids = [
            ClusterPoint(x=c[0], y=c[1], cluster=i)
            for i, c in enumerate(kmeans.cluster_centers_)
        ]

        points = [
            ClusterPoint(x=p[0], y=p[1], cluster=l)
            for p, l in zip(data_frame.values, kmeans.labels_)
        ]

        point_to_centroid = dict(zip(range(len(points)), kmeans.labels_))
        return ClusterResult(
            points=points,
            centroids=centroids,
            point_to_centroid_mappings=point_to_centroid
        )

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
