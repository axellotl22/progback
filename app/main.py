"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router, classification_decision_tree_router

app = FastAPI()

# Router für KMeans Clustering
app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
#Router für Decision Tree Classification
app.include_router(classification_decision_tree_router.router, prefix="/classification", tags=["classification"])
# Optional: Wenn die App direkt ausgeführt werden soll
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
