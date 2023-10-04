"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router
from app.routers import classification_router_decision_tree



app = FastAPI()

app.include_router(clustering_router.router, 
                   prefix="/clustering", 
                   tags=["clustering"])
app.include_router(classification_router_decision_tree.router, 
                   prefix="/classification_decision_tree", 
                   tags=["decision_tree"])
