"""
Stellt Funktionen zur Verwaltung von Jobs bereit
"""
from multiprocessing import Pool
from typing import Callable, List

import asyncio
import logging
from datetime import datetime

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job_model import JobStatus
from app.database.job_db import DBJob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_job(database: AsyncSession, user_id: str, job_parameters: str,
                     json_input: str, name: str):
    """
    Erstellt einen Job in der Datenbank
    """

    db_job = DBJob(user_id=user_id, created_at=datetime.now(),
                   status=JobStatus.WAITING, json_values=json_input,
                   job_parameters=job_parameters, job_name=name)
    database.add(db_job)
    await database.commit()
    await database.refresh(db_job)

    return db_job


async def list_jobs(database: AsyncSession, filter_user_id: str):
    """
    Gibt eine Liste von allen Jobs der Datenbank zurück
    """
    return (await database.execute(Select(DBJob).filter(DBJob.user_id == filter_user_id))).scalars()


async def list_jobs_name(database: AsyncSession, filter_user_id: str, name: str):
    """
    Gibt eine Liste von allen Jobs mit dem Namen der Datenbank zurück
    """
    return (await database.execute(Select(DBJob).filter(DBJob.user_id == filter_user_id,
                                                 DBJob.job_name == name))).scalars()


async def get_job_by_id(database: AsyncSession, job_id: int):
    """
    Gibt einen Job mit einer bestimmten ID zurück
    """
    return await database.get(DBJob, job_id)


async def get_job_by_name(database: AsyncSession, job_name: str, filter_user_id: str):
    """
    Gibt einen Job mit einer bestimmten ID zurück
    """
    return (await database.execute(Select(DBJob).filter(DBJob.user_id == filter_user_id)
                            .where(DBJob.job_name == job_name))).scalars()


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
