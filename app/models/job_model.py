"""
Enthält die Klassen für Jobs
"""
import enum
from json import dumps
from pydantic import BaseModel
from sqlalchemy import Enum


class JobStatus(enum.Enum):
    """
    Enum für den Status von Jobs
    """
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    DONE = "DONE"


class JobResponse(BaseModel):
    """
    Modell für Rückgabe aus Job-Endpunkt
    """
    job_id: int
    user_id: str
    job_name: str
    created_at: str
    status: JobStatus
    job_parameters: str


class JobResponseFull(JobResponse):
    """
    erweitertes Modell für Rückgabe aus Job-Endpunkt
    """
    json_values: str


# pylint: disable=too-few-public-methods
class UserJob:
    """
    Ein Job, welcher ausgeführt werden kann
    """

    # pylint: disable=too-many-ancestors
    class Type(Enum):
        """
        Enum für Jobtypen
        """
        KMEANS = "KMEANS"
        DECISIONTREES = "DECISIONTREES"

    def __init__(self, jobtype, parameters):
        """
        Konstruktur
        """
        self.jobtype = jobtype
        self.parameters = parameters

    def to_json(self):
        """
        Klasse als JSON
        """
        return dumps(self, default=lambda o: o.__dict__,
                     sort_keys=False, indent=None)

    jobtype: Type
    parameters: dict
