from multiprocessing import Pool
from typing import Callable, List
import asyncio
import logging
import uuid
from app.models.job_model import JobStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

jobs = []


class Job:
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
    return await Job(func, args).run_async()
