import enum

from pydantic import BaseModel
from app.services.database_service import DBBase
from sqlalchemy import Column, Integer, String, DateTime, Enum
from json import dumps
from typing import List


class JobStatus(enum.Enum):
    CANCELED = -2
    ERROR = -1
    WAITING = 0
    RUNNING = 1
    DONE = 2


class DBJob(DBBase):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    file_name = Column(String, nullable=True)
    file_hash = Column(String, nullable=True)
    status = Column(Enum(JobStatus), nullable=False)
    job_type = Column(String, nullable=False)
    result = Column(String, nullable=True)


class UserJob:
    class Type(Enum):
        KMEANS = 0
        DECISIONTREES = 1

    def __init__(self, type, parameters):
        self.type = type
        self.parameters = parameters

    def toJSON(self):
        return dumps(self, default=lambda o: o.__dict__,
              sort_keys=False, indent=None)

    type: Type
    parameters: dict


class JobEntry(BaseModel):
    """
    Ein Job als pydantic Modell für die Ausgabe über die API
    Attribute:
    - id (int): einzigartige ID des Jobs
    - user_id (userid): ID des Benutzers, welcher den Job erstellt hat
    - created_at (DateTime): Zeitpunkt der Erstellung des Jobs
    - file_name (str): Name der Datei, welche bearbeitet wird/wurde
    - file_hash (str): Hash der Datei
    - status (JobStatus): aktueller Status des Jobs
    - result (object): das Ergebnis, falls der Task fertig ist
    """
    id: int = 0
    user_id: int = 0
    created_at: str = None
    file_name: str = None
    file_hash: str = None
    status: JobStatus = 0
    job_type: str = None
    result: str = None
