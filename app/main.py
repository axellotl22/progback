"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router
from app.routers import job_router
from app.services.database_service import DBBase, engine
from app.database.user_db import create_db_and_tables
from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users
from app.routers import job_router
from app.services.database_service import DBBase, engine

app = FastAPI()

app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(job_router.router, prefix="/jobs", tags=["jobs"])

DBBase.metadata.create_all(bind=engine)
app.include_router(job_router.router, prefix="/jobs", tags=["jobs"])

# Authentication
app.include_router(fastapi_users.get_auth_router(auth_backend), tags=["auth"])
app.include_router(fastapi_users.get_register_router(UserRead, UserCreate), tags=["auth"])
app.include_router(fastapi_users.get_reset_password_router(), tags=["auth"])
app.include_router(fastapi_users.get_verify_router(UserRead), tags=["auth"])
app.include_router(fastapi_users.get_users_router(
    UserRead, UserUpdate),
    tags=["users"],
    prefix="/users"
)


@app.on_event("startup")
async def on_startup():
    """ Events on Startup """
    await create_db_and_tables()

DBBase.metadata.create_all(bind=engine)
