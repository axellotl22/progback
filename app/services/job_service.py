"""
Stellt Funktionen zur Verwaltung von Jobs bereit
"""
from multiprocessing import Pool
from typing import Callable, List

import asyncio
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.job_model import JobStatus
from app.database.job_db import DBJob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_job(database: Session, user_id: int, job_parameters: str, json_input: str):
    """
    Erstellt einen Job in der Datenbank
    """

    db_job = DBJob(user_id=user_id, created_at=datetime.now(),
                   status=JobStatus.WAITING, json_values=json_input,
                   job_parameters=job_parameters)
    database.add(db_job)
    database.commit()
    database.refresh(db_job)
    return db_job


def set_job_result(database: Session, job_id: int, result: str):
    """
    Setzt das Feld "result" für einen Job in der Datenbank
    """
    database.query(DBJob).filter(DBJob.id.is_(job_id)).update({DBJob.result: result})
    database.commit()


def list_jobs(database: Session, skip: int = 0, limit: int = 100):
    """
    Gibt eine Liste von allen Jobs der Datenbank zurück
    """
    return database.query(DBJob).offset(skip).limit(limit).all()


def get_job_by_id(database: Session, job_id: int):
    """
    Gibt einen Job mit einer bestimmten ID zurück
    """
    return database.query(DBJob).get(job_id)


def get_job_by_name(database: Session, job_name: str):
    """
    Gibt einen Job mit einer bestimmten ID zurück
    """
    return database.query(DBJob).where(DBJob.job_name.is_(job_name)).first()


class RunJob:
    """
    Führt einen Job aus
    """

    def __init__(self, func: Callable, args: List):
        self.status = JobStatus.WAITING
        self._func = func
        self._args = args
        self.result = None
        # pylint: disable=consider-using-with
        self._pool: Pool = Pool(processes=1)

    def __del__(self):
        self._pool.close()

    async def run_async(self):
        """
        Führt den Job asynchron durch
        """
        logger.info("Running \"%s\"", self._func.__name__)
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        print(self._func.__name__ + f"({self._args})")

        def _on_done(obj):
            """
            Callback, nachdem der Job abgeschlossen wurde
            """
            loop.call_soon_threadsafe(fut.set_result, obj)
            self.status = JobStatus.DONE
            self.result = obj

        def _on_error(err):
            """
            Callback, nachdem ein Fehler aufgetreten ist
            """
            loop.call_soon_threadsafe(fut.set_exception, err)
            self.status = JobStatus.ERROR

        self._pool.apply_async(func=self._func, args=self._args, callback=_on_done,
                               error_callback=_on_error)

        self.status = JobStatus.RUNNING
        return await fut

    def cancel(self):
        """
        Laufenden Job abbrechen
        """
        logger.info("Canceling \"%s\"", self._func.__name__)
        self._pool.close()
        self._pool.terminate()


async def run_job_async(func: Callable, args: List):
    """
    Simple helper-function for easier usage of RunJob
    """
    return await RunJob(func, args).run_async()
