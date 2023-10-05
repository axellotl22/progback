"""
Hauptmodul der App.
"""

from fastapi import FastAPI

from app.database import user_db, job_db
from app.routers import clustering_router
from app.routers import classification_router_decision_tree


from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users
from app.routers import job_router

app = FastAPI()

app.include_router(clustering_router.router,
                   prefix="/clustering",
                   tags=["clustering"])
app.include_router(classification_router_decision_tree.router,
                   prefix="/classification_decision_tree",
                   tags=["decision_tree"])
app.include_router(clustering_router.router, prefix="/clustering", tags=["clustering"])
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
    await user_db.create_db_and_tables()
    await job_db.create_db_and_tables()
