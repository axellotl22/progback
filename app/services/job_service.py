from multiprocessing import Pool
from typing import Callable, List
import asyncio
import logging
import uuid
from app.models.job_model import JobStatus, DBJob
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi import Depends
from app.services.database_service import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

jobs = []


def create_job(db: Session, user_id: int, file_name: str, file_hash: str, job_type: str):
    db_job = DBJob(user_id=user_id, created_at=datetime.now(), status=JobStatus.WAITING,
                   file_name=file_name, file_hash=file_hash, job_type=job_type)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def set_job_result(db: Session, job_id: int, result: str):
    db.query(DBJob).filter(DBJob.id == job_id).update({DBJob.result: result})
    db.commit()
    return


def list_jobs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBJob).offset(skip).limit(limit).all()


def get_job_by_id(db: Session, job_id: int):
    return db.query(DBJob).filter(DBJob.id == job_id).first()



class RunJob:
    """
    Test
    """

    def __init__(self, func: Callable, args: List):
        jobs.append(self)
        self.status = JobStatus.WAITING
        self.uuid = uuid.uuid1()
        self._func = func
        self._args = args
        self.result = None
        self._pool: Pool = Pool(processes=1)

    async def run_async(self):
        logger.info("Running \"%s\" (%s)", self._func.__name__, self.uuid)
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def _on_done(obj):
            loop.call_soon_threadsafe(fut.set_result, obj)
            self.status = JobStatus.DONE
            self.result = obj

        def _on_error(err):
            loop.call_soon_threadsafe(fut.set_exception, err)
            self.status = JobStatus.ERROR

        self._pool.apply_async(func=self._func, args=self._args, callback=_on_done, error_callback=_on_error)
        self.status = JobStatus.RUNNING
        return await fut

    def cancel(self):
        logger.info("Canceling \"%s\" (%s)", self._func.__name__, self.uuid)
        self._pool.close()
        self._pool.terminate()


async def run_job_async(func: Callable, args: List):
    return await RunJob(func, args).run_async()
