"""
Definition von Job Endpoints
"""
import asyncio
import logging
import uuid
from typing import Optional, Union, List
import json

from fastapi import (APIRouter, WebSocket, WebSocketDisconnect,
                     Depends, UploadFile, File, Query, HTTPException)
from sqlalchemy.orm import Session
from pandas import read_json

from app.models.clustering_model import ClusterResult
from app.services.clustering_algorithms import CustomKMeans
from app.services.clustering_service import process_and_cluster
from app.database.connection import get_db
from app.services.job_service import (RunJob, list_jobs, create_job, get_job_by_id,
                                      get_job_by_name, list_jobs_name)
from app.models.job_model import UserJob, JobStatus, JobResponse, JobResponseFull
from app.services.utils import save_temp_file, delete_file, load_dataframe
from app.database.user_db import User
from app.entitys.user import active_user, auth_backend, UserManager, get_user_manager

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


# pylint: disable=too-many-arguments
def kmeans_job(job_id, data_frame, name, columns, k_cluster, distance_metric,
               cluster_count_determination):
    """
    Führt KMeans durch
    :param data_frame:
    :param name: festlegen eines Namens
    :param columns: Spaltennamen
    :param k_cluster: Clusteranzahl
    :param distance_metric: Distanzmethode
    :param cluster_count_determination: Clusterbestimmungsmethode
    :return: Ergebnis
    """

    # Create file for method
    with open(f"{TEMP_FILES_DIR}{uuid.uuid1()}.json", "w", encoding="utf-8") as outfile:
        outfile.write(data_frame.to_json())
        filename = outfile.name

    results = process_and_cluster(data_frame, cluster_count_determination, distance_metric,
                                  columns, k_cluster, filename)

    # Determine method used
    if k_cluster:
        used_method = f"Manually set to {k_cluster}"
    else:
        used_method = cluster_count_determination

    # Return clustering result model
    return ClusterResult(
        user_id=-1,
        request_id=job_id,
        name=name or "KMeans",
        # pylint: disable=duplicate-code
        cluster=results["cluster"],
        x_label=results["x_label"],
        y_label=results["y_label"],
        iterations=results["iterations"],
        used_distance_metric=distance_metric,
        used_optK_method=used_method,
        clusters_elbow=results["clusters_elbow"],
        clusters_silhouette=results["clusters_silhouette"]
    )


@router.websocket("/{job_id}/")
async def job_websocket(web_socket: WebSocket, job_id: int, database: Session = Depends(get_db),
                        user_manager: UserManager = Depends(get_user_manager)):
    """
    Web-Socket für Jobs
    """

    await web_socket.accept()

    # Auth
    cookie = web_socket.cookies.get("fastapiusersauth")
    user = await auth_backend.get_strategy().read_token(cookie, user_manager)
    if not user or not user.is_active:
        await web_socket.send_text("Unauthorized")
        await web_socket.close()
        return

    # Find job in database
    current_job = get_job_by_id(database, job_id)
    if current_job is None or current_job.user_id != str(user.id):
        await web_socket.send_json({})
        await web_socket.close()
        return

    if current_job.status is JobStatus.DONE:
        await web_socket.send_json(current_job.json_values)
        await web_socket.close()
        return

    if current_job.status is JobStatus.ERROR:
        await web_socket.send_json({"status": "error"})
        await web_socket.close()
        return

    if current_job.status is JobStatus.RUNNING:
        await web_socket.send_json({"status": "Job läuft bereits!"})
        await web_socket.close()
        return

    # bei CANCELED und WAITING Job starten
    current_job.status = JobStatus.RUNNING
    database.commit()

    # Load job info (function and parameters) from database
    job_json = json.loads(current_job.job_parameters)
    job_type = UserJob(job_json["jobtype"], job_json["parameters"])

    # Run Job
    j = RunJob(func=kmeans_job,
               args=[current_job.id,
                     read_json(current_job.json_values), current_job.job_name]
                    + list(job_type.parameters.values()))
    run_job = asyncio.ensure_future(j.run_async())

    # Also check if client disconnects
    run_check = asyncio.ensure_future(is_disconnected(web_socket))

    # Wait for either completion or user disconnect
    await asyncio.wait([run_job, run_check], return_when=asyncio.FIRST_COMPLETED)

    # If disconnect -> cancel job
    if run_check.done() and run_check:
        j.cancel()
        database.commit()
        return

    result = j.result.model_dump()

    current_job.status = j.status
    current_job.json_values = str(result)

    database.commit()
    database.refresh(current_job)

    # Send result as websocket message in json format
    await web_socket.send_json(result)
    await web_socket.close()


# pylint: disable=too-many-arguments too-many-locals
@router.post("/create/kmeans", response_model=JobResponse)
async def create_kmeans_job(
        database: Session = Depends(get_db),
        user: User = Depends(active_user),
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
        ),
        name: str = "KMeans"
):
    """
    Erstellt einen Job für KMeans
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
    file_path = save_temp_file(file, TEMP_FILES_DIR)

    # Create Job in database and return object as response
    job_type = UserJob(jobtype=UserJob.Type.KMEANS,
                       parameters={"columns": columns,
                                   "k_cluster": k_cluster,
                                   "distance_metric": distance_metric,
                                   "cluster_count_determination": cluster_count_determination})
    try:
        data_frame = load_dataframe(file_path)

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        delete_file(file_path)

    db_job = create_job(database, user_id=user.id, job_parameters=job_type.to_json(),
                        json_input=data_frame.to_json(), name=name)

    return JobResponse(job_id=db_job.id,
                       user_id=db_job.user_id,
                       job_name=db_job.job_name or "KMeans",
                       created_at=str(db_job.created_at),
                       job_parameters=db_job.job_parameters,
                       status=db_job.status
                       )


@router.get("/list/", response_model=Union[List[JobResponse], List[JobResponseFull]])
async def job_list(job_id: Optional[int] = None, job_name: Optional[str] = None,
                   with_values: Optional[bool] = False, database: Session = Depends(get_db),
                   user: User = Depends(active_user)):
    """
    Gibt alle Jobs in der Datenbank als Json zurück. Optional
    können einzelne Jobs mit bestimmter ID
    oder Namen zurückgegeben werden. Mit dem Parameter
    with_values werden auch gespeicherte Daten (berechnete
    Ergebnisse oder Rohdaten) zurückgegeben.
    """

    if job_id is not None:
        db_job = get_job_by_id(database=database, job_id=job_id)
        if db_job is None or db_job.user_id != str(user.id):
            raise HTTPException(status_code=404, detail="Job nicht gefunden!")
        list_content = [db_job, ]
    elif job_name is not None:
        db_job = get_job_by_name(database, job_name, str(user.id))
        if db_job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden!")
        list_content = list_jobs_name(database, str(user.id), job_name)
    else:
        list_content = list_jobs(database, str(user.id))

    if with_values:
        result = [JobResponseFull(job_id=db_job.id,
                                  user_id=db_job.user_id,
                                  job_name=db_job.job_name or "KMeans",
                                  created_at=str(db_job.created_at),
                                  job_parameters=db_job.job_parameters,
                                  status=db_job.status,
                                  json_values=db_job.json_values
                                  ) for db_job in list_content]
    else:
        result = [JobResponse(job_id=db_job.id,
                              user_id=db_job.user_id,
                              job_name=db_job.job_name or "KMeans",
                              created_at=str(db_job.created_at),
                              job_parameters=db_job.job_parameters,
                              status=db_job.status
                              ) for db_job in list_content]

    return result
