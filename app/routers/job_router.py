import asyncio
import math
from typing import Optional, Union
import json
from types import SimpleNamespace

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Query
from sqlalchemy.orm import Session

from app.services.clustering_algorithms import CustomKMeans
from app.services.database_service import get_db
from app.services.job_service import RunJob, list_jobs, create_job, get_job_by_id
from app.models.job_model import UserJob
from app.routers.clustering_router import perform_kmeans_clustering

router = APIRouter()


def background():
    print(f"Task gestartet!: {id}")
    data = [math.sqrt(i) for i in range(50000000)]
    print(f"Task beendet!: {id}")
    return "Fertig!"


async def is_disconnected(socket: WebSocket):
    try:
        print("Running disconnect check!")
        await socket.receive()
        return False
    except WebSocketDisconnect:
        print("Websocket exception!")
        return True


@router.websocket("/{job_id}/")
async def job_websocket(web_socket: WebSocket, job_id: int, db: Session = Depends(get_db)):
    print("connection!")

    current_job = get_job_by_id(db, job_id)
    if current_job is None:
        await web_socket.close()
        return

    await web_socket.accept()
    job_json = json.loads(current_job.job_type)
    job_type = UserJob(job_json["type"], job_json["parameters"])
    print(job_type)
    j = RunJob(func=perform_kmeans_clustering, args=job_type.parameters)
    print(job_type.parameters)
    run_job = asyncio.ensure_future(j.run_async())
    run_check = asyncio.ensure_future(is_disconnected(web_socket))
    await asyncio.wait([run_job, run_check], return_when=asyncio.FIRST_COMPLETED)
    if run_check.done() and run_check:
        print("Client disconnected!")
        current_job.cancel()
        return

    await web_socket.send(run_job.result())
    await web_socket.close()


@router.post("/create/kmeans")
async def create_kmeans_job(
        db: Session = Depends(get_db),
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
    job_type = UserJob(type=UserJob.Type.KMEANS,
                       parameters={"columns": [column1, column2],
                                   "file_name": file.filename, "k_cluster": k_cluster,
                                   "distance_metric": distance_metric,
                                   "cluster_count_determination": cluster_count_determination})

    return create_job(db, user_id=0, file_name=file.filename, job_type=job_type.toJSON(), file_hash="")


@router.get("/list/")
async def job_list(db: Session = Depends(get_db)):
    return list_jobs(db)
