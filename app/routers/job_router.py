import asyncio
import math

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.job_service import Job, jobs, run_job_async, JobStatus
from app.models.job_model import JobList, JobEntry, JobStatus

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
async def job_websocket(web_socket: WebSocket, job_id: str):
    print("connection!")
    current_job = None
    for job in jobs:
        if str(job.uuid) == job_id:
            print("job found")
            current_job = job

    if current_job is None:
        await web_socket.close()
        return

    await web_socket.accept()
    print("accepted")
    run_job = asyncio.ensure_future(current_job.run_async())
    run_check = asyncio.ensure_future(is_disconnected(web_socket))
    await asyncio.wait([run_job, run_check], return_when=asyncio.FIRST_COMPLETED)
    if run_check.done() and run_check:
        print("Client disconnected!")
        current_job.cancel()
        return

    await web_socket.send_json(run_job.result())
    await web_socket.close()


@router.get("/test/")
async def job_test():
    job = Job(background, [])
    return await job.run_async()


@router.get("/list/")
async def job_list():
    jobList = JobList()
    for job in jobs:
        jobList.jobs.append(JobEntry(uuid=str(job.uuid), status=job.status, result=job.result))
    return jobList
