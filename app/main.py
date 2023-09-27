"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router
from app.routers import job_router
from app.services.database_service import DBBase, engine

app = FastAPI()


app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(job_router.router, prefix="/jobs", tags=["jobs"])

DBBase.metadata.create_all(bind=engine)