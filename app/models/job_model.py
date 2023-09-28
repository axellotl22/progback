"""
Enthält die Klassen für Jobs
"""
import enum
from json import dumps

from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Enum
from app.services.database_service import DBBase


class JobStatus(enum.Enum):
    """
    Enum für den Status von Jobs
    """
    CANCELED = -2
    ERROR = -1
    WAITING = 0
    RUNNING = 1
    DONE = 2


# pylint: disable=too-few-public-methods
class DBJob(DBBase):
    """
    Job als Modell für SQLAlchemy
    """
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
    """
    Ein Job, welcher ausgeführt werden kann
    """

    # pylint: disable=too-many-ancestors
    class Type(Enum):
        """
        Enum für Jobtypen
        """
        KMEANS = 0
        DECISIONTREES = 1

    def __init__(self, jobtype, parameters):
        """
        Konstruktur
        :param jobtype:
        :param parameters:
        """
        self.jobtype = jobtype
        self.parameters = parameters

    def to_json(self):
        """
        Klasse als JSON
        :return:
        """
        return dumps(self, default=lambda o: o.__dict__,
                     sort_keys=False, indent=None)

    jobtype: Type
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
