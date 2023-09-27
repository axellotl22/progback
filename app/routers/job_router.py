import asyncio
import datetime
import math
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.services.job_service import jobs
from app.models.job_model import JobStatus, DBJob
from app.services.database_service import get_db
from sqlalchemy.orm import Session
router = APIRouter()


def create_job(db: Session):
    db_job = DBJob(user_id=0, created_at=datetime.now(), status=JobStatus.WAITING)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def list_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBJob).offset(skip).limit(limit).all()


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
            current_job = job

    if current_job is None:
        await web_socket.close()
        return

    await web_socket.accept()
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
async def job_test(db: Session = Depends(get_db)):
    return create_job(db)


@router.get("/list/")
async def job_list(db: Session = Depends(get_db)):
    return list_jobs(db)
