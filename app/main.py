"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router

app = FastAPI()

app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
