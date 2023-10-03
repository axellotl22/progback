"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router, elbow_router, basic_kmeans_router
from app.routers import user_router


app = FastAPI()


app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(user_router.router, prefix="/user", tags=["user"])

app.include_router(elbow_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(basic_kmeans_router.router, prefix="/clustering", tags=["clustering"])