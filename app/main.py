"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router
from app.entitys.user import (create_user)

app = FastAPI()

create_user()

app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])

# Optional: Wenn die App direkt ausgeführt werden soll
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
