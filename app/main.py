"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router
from app.routers import classification_router_decision_tree
from app.routers import user_router


app = FastAPI()


app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(classification_router_decision_tree.router, prefix="/classification_decision_tree", tags=["decision_tree"])
app.include_router(user_router.router, prefix="/user", tags=["user"])
