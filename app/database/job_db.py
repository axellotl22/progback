"""
Datenbank-Komponente für Jobs
"""
from sqlalchemy import Column, String, Integer, DateTime, Enum, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from app.database.connection import get_async_engine
from app.models.job_model import JobStatus

Base = declarative_base()

# pylint: disable=too-few-public-methods
class DBJob(Base):
    """
    Job als Modell für SQLAlchemy
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    job_name = Column(String(255), nullable=True)
    status = Column(Enum(JobStatus), nullable=False)
    job_parameters = Column(String(5000), nullable=False)
    json_values = Column(Text, nullable=True)


engine = get_async_engine()
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_db_and_tables():
    """
    Erstellt die initale Datenbank
    :return:
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
