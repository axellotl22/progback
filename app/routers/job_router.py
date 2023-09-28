"""
Definition von Job Endpoints
"""
import asyncio
import logging
import os
from typing import Optional, Union
import json

from fastapi import (APIRouter, WebSocket, WebSocketDisconnect,
                     Depends, UploadFile, File, Query, HTTPException)
from sqlalchemy.orm import Session

from app.models.clustering_model import ClusterResult
from app.services.clustering_algorithms import CustomKMeans
from app.services.clustering_service import process_and_cluster
from app.services.database_service import get_db
from app.services.job_service import RunJob, list_jobs, create_job, get_job_by_id
from app.models.job_model import UserJob, JobStatus
from app.services.utils import save_temp_file, delete_file, load_dataframe

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TEMP_FILES_DIR = "temp_files/"


async def is_disconnected(socket: WebSocket):
    """
    Checkt, ob der Client im Websocket disconnected
    :param socket: offener Socket
    :return: True bei Disconnect, bei Nachricht false
    """
    try:
        print("Running disconnect check!")
        await socket.receive()
        return False
    except WebSocketDisconnect:
        print("WebsocketDisconnectException!")
        return True


def kmeans_job(file_name, columns, k_cluster, distance_metric, cluster_count_determination):
    """
    Führt KMeans durch
    :param file_name: Dateiname
    :param columns: Spaltennamen
    :param k_cluster: Clusteranzahl
    :param distance_metric: Distanzmethode
    :param cluster_count_determination: Clusterbestimmungsmethode
    :return: Ergebnis
    """
    file_path = TEMP_FILES_DIR + f"/{file_name}"

    try:
        data_frame = load_dataframe(file_path)

        results = process_and_cluster(data_frame, cluster_count_determination, distance_metric,
                                      columns, k_cluster)

        # Determine method used
        if k_cluster:
            used_method = f"Manually set to {k_cluster}"
        else:
            used_method = cluster_count_determination

        # pylint: disable=duplicate-code
        # Return clustering result model
        return ClusterResult(
            user_id=0,
            request_id=0,
            name=f"K-Means Result for: {os.path.splitext(file_name)[0]}",
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
        delete_file(file_path)


@router.websocket("/{job_id}/")
async def job_websocket(web_socket: WebSocket, job_id: int, database: Session = Depends(get_db)):
    """
    Web-Socket für Jobs
    :param web_socket:
    :param job_id:
    :param db:
    :return:
    """
    print("connection!")

    # Find job in database
    current_job = get_job_by_id(database, job_id)
    if current_job is None or current_job.status is not JobStatus.WAITING:
        await web_socket.close()
        return

    await web_socket.accept()

    # Load job info (function and parameters) from database
    job_json = json.loads(current_job.job_type)
    job_type = UserJob(job_json["jobtype"], job_json["parameters"])

    # Run Job
    j = RunJob(func=kmeans_job, args=list(job_type.parameters.values()))
    run_job = asyncio.ensure_future(j.run_async())

    # Also check if client disconnects
    run_check = asyncio.ensure_future(is_disconnected(web_socket))

    # Wait for either completion or user disconnect
    await asyncio.wait([run_job, run_check], return_when=asyncio.FIRST_COMPLETED)

    # If disconnect -> cancel job
    if run_check.done() and run_check:
        print("Client disconnected!")
        current_job.cancel()
        return

    result = j.result.model_dump()

    current_job.status = j.status
    current_job.result = str(result)

    database.commit()
    database.refresh(current_job)

    # Send result as websocket message in json format
    await web_socket.send_json(result)
    await web_socket.close()

# pylint: disable=too-many-arguments
@router.post("/create/kmeans")
async def create_kmeans_job(
        database: Session = Depends(get_db),
        file: UploadFile = File(...),
        column1
        : Optional[Union[str, int]] = None,
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
        )
):
    """
    Erstellt einen Job für KMeans
    :param database:
    :param file:
    :param column1:
    :param column2:
    :param k_cluster:
    :param distance_metric:
    :param cluster_count_determination:
    :return:
    """
    # pylint: disable=duplicate-code
    # Validate distance metric
    supported_metrics = list(CustomKMeans.supported_distance_metrics.keys())
    if distance_metric not in supported_metrics:
        error_msg = (
            f"{distance_metric}: Invalid distance metric. Supported metrics are: "
            f"{', '.join(supported_metrics)}"
        )
        raise HTTPException(400, error_msg)

    # Convert columns to int if given as string
    if isinstance(column1, str):
        column1 = int(column1)
    if isinstance(column2, str):
        column2 = int(column2)

    # Process file
    columns = [column1, column2] if column1 and column2 else None
    save_temp_file(file, TEMP_FILES_DIR)

    # Create Job in database and return object as response
    job_type = UserJob(jobtype=UserJob.Type.KMEANS,
                       parameters={"file_name": file.filename, "columns": columns,
                                   "k_cluster": k_cluster,
                                   "distance_metric": distance_metric,
                                   "cluster_count_determination": cluster_count_determination})

    return create_job(database, user_id=0, file_name=file.filename,
                      job_type=job_type.to_json(), file_hash="")


@router.get("/list/")
async def job_list(database: Session = Depends(get_db)):
    """
    Gibt alle Jobs in der Datenbank als Json zurück
    :param db:
    :return:
    """
    return list_jobs(database)
