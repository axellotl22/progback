"""
Definition von Job Endpoints
"""
import asyncio
import logging
import os
from typing import Optional, Union, List
import json

from fastapi import (APIRouter, WebSocket, WebSocketDisconnect,
                     Depends, UploadFile, File, Query, HTTPException)
from sqlalchemy.ext.asyncio import AsyncSession
from pandas import read_json

from app.database.connection import get_async_db
from app.services.custom_kmeans import BaseOptimizedKMeans
from app.services.job_service import (RunJob, list_jobs, create_job, get_job_by_id,
                                      get_job_by_name, list_jobs_name)
from app.services import basic_kmeans_service
from app.models.job_model import UserJob, JobStatus, JobResponse, JobResponseFull
from app.services.utils import process_uploaded_file
from app.database.user_db import User
from app.entitys.user import active_user, auth_backend, UserManager, get_user_manager

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def is_disconnected(socket: WebSocket):
    """
    Function that checks if the client is disconnected
    :param socket: current WebSocket
    :return: True on Disconnect, False on incoming message
    """
    try:
        logger.info("Running disconnect check!")
        await socket.receive()
        return False
    except WebSocketDisconnect:
        logger.warning("WebSocket-Client disconnected!")
        return True


def run_job_22(json_values, job_parameters, name, job_id, user_id):
    """
    Runs the job
    """
    # Load job info (function and parameters) from database
    job_parameters = json.loads(job_parameters)
    job_type = UserJob(job_parameters["jobtype"], job_parameters["parameters"])

    # check jobType
    if job_type.jobtype == UserJob.Type.BASIC_2d_KMEANS:
        return basic_kmeans_service.perform_kmeans_from_dataframe(
            data_frame=read_json(json_values),
            filename=name,
            distance_metric=job_type.parameters[
                "distance_metric"],
            kmeans_type=job_type.parameters["kmeans_type"],
            user_id=user_id,
            request_id=job_id,
            advanced_k=job_type.parameters["k_clusters"],
            normalize=job_type.parameters["normalize"])
    # currently only Basic_2d_KMEANS is supported
    return None


# pylint: disable=too-many-statements
@router.websocket("/{job_id}/")
async def job_websocket(web_socket: WebSocket, job_id: int,
                        database: AsyncSession = Depends(get_async_db),
                        user_manager: UserManager = Depends(get_user_manager)):
    """
    Web-Socket for job connections
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
    current_job = await get_job_by_id(database, job_id)

    # check if correct user
    if current_job is None or current_job.user_id != str(user.id):
        await web_socket.send_json({})
        await web_socket.close()
        return

    # disallow specific JobStatus values
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

    # run if JobStatus is CANCELED or WAITING
    current_job.status = JobStatus.RUNNING
    await database.commit()
    await database.refresh(current_job)

    # Run Job
    try:

        # run in sync for test_mode
        if os.getenv("TEST_MODE") == "True" or os.getenv("TEST_MODE") is None:
            result = run_job_22(current_job.json_values, current_job.job_parameters,
                                current_job.job_name, current_job.id, 0).model_dump()

        # run async if not
        else:
            j = RunJob(func=run_job_22,
                       args=[current_job.json_values,
                             current_job.job_parameters,
                             current_job.job_name,
                             current_job.id,
                             0])
            run_job = asyncio.ensure_future(j.run_async())

            # Also check if client disconnects
            run_check = asyncio.ensure_future(is_disconnected(web_socket))

            # Wait for either completion or user disconnect
            await asyncio.wait([run_job, run_check], return_when=asyncio.FIRST_COMPLETED)

            # If disconnect -> cancel job
            if run_check.done() and run_check:
                j.cancel()
                await database.commit()
                return
            await run_job
            result = j.result.model_dump()
            current_job.status = j.status

        result["user_id"] = current_job.user_id

        current_job.json_values = str(result)

    # Handle errors
    except ValueError as ex:
        await web_socket.send_json({"error": "Unsupported file type"})
        await web_socket.close()
        current_job.status = JobStatus.ERROR
        await database.commit()
        raise ex
    except Exception as ex:
        await web_socket.send_json({"error": "Processing error"})
        await web_socket.close()
        current_job.status = JobStatus.ERROR
        await database.commit()
        raise ex

    await database.commit()
    await database.refresh(current_job)

    # Send result as websocket message in json format
    await web_socket.send_json(result)
    await web_socket.close()


@router.get("/list/", response_model=Union[List[JobResponse], List[JobResponseFull]])
async def job_list(job_id: Optional[int] = None,
                   job_name: Optional[str] = None,
                   with_values: Optional[bool] = False,
                   database: AsyncSession = Depends(get_async_db),
                   user: User = Depends(active_user)):
    """
    Returns all jobs from the current user in the database as json list. Allows filtering by
    "job_id" and "job_name". If the parameter "with_values" is true, the results on done
    jobs and the input on pending jobs is also returned. Logging in is necessary to use
    this endpoint.

    Args:

    job_id (int): Optional query parameter for specific job by id
    job_name (str): Optional query parameter for specific job(s) by name
    with_values (bool): Optional also return the results/inputs of the job

    Returns:

    List(JobResponse): List containing information about the jobs.

    """

    if job_id is not None:
        db_job = await get_job_by_id(database=database, job_id=job_id)
        if db_job is None or db_job.user_id != str(user.id):
            raise HTTPException(status_code=404, detail="Job nicht gefunden!")
        list_content = [db_job, ]
    elif job_name is not None:
        db_job = await get_job_by_name(database, job_name, str(user.id))
        if db_job is None:
            raise HTTPException(status_code=404, detail="Job nicht gefunden!")
        list_content = await list_jobs_name(database, str(user.id), job_name)
    else:
        list_content = await list_jobs(database, str(user.id))

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

# pylint: disable=too-many-arguments
@router.post("/create/basic_2d_kmeans")
async def kmeans_job2(
        database: AsyncSession = Depends(get_async_db),
        user: User = Depends(active_user),
        file: UploadFile = File(...),
        name: str = Query("KMeans", description="Name of the job"),
        column1: int = Query(0,
                             description="Index of the first column"),
        column2: int = Query(1,
                             description="Index of the second column"),
        distance_metric: str = Query(
            "EUCLIDEAN",
            description="/".join(BaseOptimizedKMeans.supported_distance_metrics.keys())),
        kmeans_type: str = Query("OptimizedKMeans",
                                 description="OptimizedKMeans/OptimizedMiniBatchKMeans"),
        k_clusters: int = Query(2, description="Number of clusters"),
        normalize: bool = True
):
    """
    Creates a job for basic 2D KMeans

    The job is created with status WAITING returns a job_id. To run the job a websocket connection
    to "/jobs/{job_id}/" is required. After the job is done the result will be sent as a json
    object on that connection. The websocket requires an authentication. For that the cookie
    "fastapiusersauth" is necessary. It is created by logging-in to the api. Jobs created here
    are added to the job history. Logging in is necessary to use this endpoint.

    Args:

    file (UploadFile): Dataset uploaded by the user for clustering.
    name (str): Choose a name for the job
    column_1, column_2 (int): Indices of columns to be used for 2D clustering.
    distance_metric (str): Selected metric for measuring distances between data points.
    kmeans_type (str): Algorithm variant for clustering. 'OptimizedKMeans' is conventional,
    while 'OptimizedMiniBatchKMeans' is faster but approximative.
    k_clusters (int): Desired number of clusters.
    normalize (bool): Whether to normalize data before clustering. Default is True.

    Returns:

    JobResponse: Object containing information about the created job.

    """

    data_frame, _ = process_uploaded_file(file, [column1, column2])
    json_input = data_frame.to_json()

    job_type = UserJob(jobtype=UserJob.Type.BASIC_2d_KMEANS,
                       parameters={"k_clusters": k_clusters,
                                   "distance_metric": distance_metric,
                                   "kmeans_type": kmeans_type,
                                   "normalize": normalize})

    db_job = await create_job(database=database, user_id=str(user.id),
                              job_parameters=job_type.to_json(),
                              json_input=json_input, name=name)

    return JobResponse(job_id=db_job.id,
                       user_id=db_job.user_id,
                       job_name=db_job.job_name or "KMeans",
                       created_at=str(db_job.created_at),
                       job_parameters=db_job.job_parameters,
                       status=db_job.status
                       )
