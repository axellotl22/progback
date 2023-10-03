"""
Hauptmodul der App.
"""

from fastapi import FastAPI
<<<<<<< HEAD
from app.routers import clustering_router, elbow_router, basic_kmeans_router
from app.routers import user_router

=======
from app.routers import clustering_router
from app.database.user_db import create_db_and_tables
from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users
>>>>>>> 2401ac48a8d627545bd50dc279b95f61a0ab8b09

app = FastAPI()

app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
<<<<<<< HEAD
app.include_router(user_router.router, prefix="/user", tags=["user"])

app.include_router(elbow_router.router, prefix="/clustering", tags=["clustering"])
app.include_router(basic_kmeans_router.router, prefix="/clustering", tags=["clustering"])
=======

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
>>>>>>> 2401ac48a8d627545bd50dc279b95f61a0ab8b09
