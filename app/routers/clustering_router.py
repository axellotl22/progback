"""Router for clustering endpoints."""

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
from app.services.job_service import run_job_async, Job, jobs
from app.models.job_model import JobEntry, JobStatus

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()


@router.post("/perform-kmeans-clustering/", response_model=Union[ClusterResult, JobEntry])
# pylint: disable=too-many-arguments
async def perform_kmeans_clustering(
    file: UploadFile = File(...),
    column1: Optional[Union[str, int]] = None,
    column2: Optional[Union[str, int]] = None,
    k_cluster: Optional[int] = Query(
        None, alias="kCluster", description="Number of clusters"
    ),
    distance_metric: Optional[str] = Query(
        "EUCLIDEAN", alias="distanceMetric",
        description=", ".join(CustomKMeans.supported_distance_metrics.keys())
    ),
    cluster_count_determination: Optional[str] = Query(
        "ELBOW", alias="clusterDetermination",
        description="ELBOW, SILHOUETTE"
    ),
    force_run_as_job: Optional[bool] = False
):
    """
    This endpoint processes the uploaded file and returns
    the clustering results. User can optionally specify
    columns and distance metric.
    """

    # Validate distance metric
    supported_metrics = list(CustomKMeans.supported_distance_metrics.keys())
    if distance_metric not in supported_metrics:
        error_msg = (
            f"Invalid distance metric. Supported metrics are: "
            f"{', '.join(supported_metrics)}"
        )
        raise HTTPException(400, error_msg)

    # Convert columns to int if given as string
    if isinstance(column1, str):
        column1 = int(column1)
    if isinstance(column2, str):
        column2 = int(column2)

    # Determine method used
    if k_cluster:
        used_method = f"Manually set to {k_cluster}"
    else:
        used_method = cluster_count_determination

    # Process file
    columns = [column1, column2] if column1 and column2 else None
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    try:
        data_frame = load_dataframe(file_path)

        if force_run_as_job:
            job = Job(func=process_and_cluster, args=[data_frame, cluster_count_determination, distance_metric,
                                                      columns, k_cluster])
            return JobEntry(uuid=str(job.uuid), status=JobStatus.WAITING)
        else:
            results = await run_job_async(func=process_and_cluster,
                                          args=[data_frame, cluster_count_determination, distance_metric, columns,
                                                k_cluster])

        # Return clustering result model
        return ClusterResult(
            user_id=0,
            request_id=0,
            name=f"K-Means Result for: {os.path.splitext(file.filename)[0]}",
            cluster=results["cluster"],
            x_label=results["x_label"],
            y_label=results["y_label"],
            iterations=results["iterations"],
            used_distance_metric=distance_metric,
            used_optK_method=used_method,
            clusters_elbow=results["clusters_elbow"],
            clusters_silhouette=results["clusters_silhouette"]
        )

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        if not TEST_MODE and not force_run_as_job:
            delete_file(file_path)
