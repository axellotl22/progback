from pydantic import BaseModel
from enum import Enum
from typing import List


class JobStatus(Enum):
    CANCELED = -2
    ERROR = -1
    WAITING = 0
    RUNNING = 1
    DONE = 2


class JobInfo(BaseModel):
    uuid: str
    status: JobStatus


class JobEntry(BaseModel):
    """
    Ein einzelner Job

    Attribute:
    - uuid (str): einzigartige ID des Jobs
    - userid (userid): ID des Benutzers, welcher den Job erstellt hat
    - status (JobStatus): aktueller Status des Jobs
    - result (object): das Ergebnis, falls der Task fertig ist
    """
    uuid: str
    userid: int = 0
    status: JobStatus
    result: object = None


class JobList(BaseModel):
    """
    Liste von allen Jobs

    Attribute:
    - jobs (List[JobEntry]): Liste von allen Jobs
    """
    jobs: List[JobEntry] = []
