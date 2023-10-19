"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (basic_kmeans_router, 
                         elbow_router,
                         advanced_kmeans_router,
                         basic_three_d_kmeans_router,
                         advanced_three_d_kmeans_router,
                         basic_n_d_kmeans_router,
                         advanced_n_d_kmeans_router,
                         classification_router_decision_tree,
                         healthcheck_router,)

from app.database import user_db, job_db


from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users
from app.routers import job_router

app = FastAPI()

# ----------------------- Konfiguration CORS ---------------------------------------------
origins = [
    "http://localhost",
    "https://localhost",
    "https://clustericke.de",
    "https://flutter-smart-classificator-jvhasbzqea-uc.a.run.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- 2D kmeans Routers ---------------------------------------------
app.include_router(basic_kmeans_router.router, prefix="/basic", tags=["2D K-Means"])
app.include_router(advanced_kmeans_router.router, prefix="/advanced", tags=["2D K-Means"])


# ----------------------- 3D kmeans Routers ---------------------------------------------
app.include_router(basic_three_d_kmeans_router.router, prefix="/basic", tags=["3D K-Means"])
app.include_router(advanced_three_d_kmeans_router.router, prefix="/advanced", tags=["3D K-Means"])


# ----------------------- nD kmeans Routers ---------------------------------------------
app.include_router(basic_n_d_kmeans_router.router, prefix="/basic", tags=["N dimensional K-Means"])
app.include_router(advanced_n_d_kmeans_router.router, prefix="/advanced", 
                   tags=["N dimensional K-Means"])


# ----------------------- Elbow Router --------------------------------------------------
app.include_router(elbow_router.router, prefix="/determination", tags=["determination"])


# ----------------------- Classification Router -----------------------------------------
app.include_router(classification_router_decision_tree.router,
                   prefix="/classification_decision_tree",
                   tags=["decision_tree"])


# ----------------------- Job Router ----------------------------------------------------
app.include_router(job_router.router, prefix="/jobs", tags=["jobs"])


# ----------------------- Auth Router ---------------------------------------------------
app.include_router(fastapi_users.get_auth_router(auth_backend), tags=["auth"])
app.include_router(fastapi_users.get_register_router(UserRead, UserCreate), tags=["auth"])
app.include_router(fastapi_users.get_users_router(
    UserRead, UserUpdate),
    tags=["users"],
    prefix="/users"
)


# ----------------------- Health Check ---------------------------------------------------
app.include_router(healthcheck_router.router, tags=["healtcheck"])


@app.on_event("startup")
async def on_startup():
    """ Events on Startup """
    await user_db.create_db_and_tables()
    await job_db.create_db_and_tables()
