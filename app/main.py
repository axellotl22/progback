"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import clustering_router, basic_kmeans_router, elbow_router
from app.database.user_db import create_db_and_tables
from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users

app = FastAPI()

#K-Means Variation
app.include_router(clustering_router.router, prefix="/all_in_one", tags=["kmeans"])
app.include_router(basic_kmeans_router.router, prefix="/basic", tags=["kmeans"])

#Determinaton of optimal k cluster
app.include_router(elbow_router.router, prefix="/determination", tags=["determination"])


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
