"""
Hauptmodul der App.
"""

from fastapi import FastAPI
from app.routers import (all_in_one_clustering, 
                         basic_kmeans_router, 
                         elbow_router,
                         advanced_kmeans_router,
                         basic_three_d_kmeans_router,
                         advanced_three_d_kmeans_router,
                         basic_n_d_kmeans_router,
                         advanced_n_d_kmeans_router)

from app.database import user_db, job_db
from app.routers import classification_router_decision_tree


from app.entitys.user import UserCreate, UserRead, UserUpdate, auth_backend, fastapi_users
from app.routers import job_router

app = FastAPI()

# ----------------------- K-Means Router ------------------------------------------------
app.include_router(all_in_one_clustering.router, prefix="/all_in_one", tags=["kmeans"])



# ----------------------- 2D kmeans Routers ---------------------------------------------
app.include_router(basic_kmeans_router.router, prefix="/basic", tags=["2D kmeans"])
app.include_router(advanced_kmeans_router.router, prefix="/advanced", tags=["2D kmeans"])

# This comment can be removed once the frontend team has taken over the new routers.
app.include_router(all_in_one_clustering.router, prefix="/clustering", tags=["clustering"])



# ----------------------- 3D kmeans Routers ---------------------------------------------
app.include_router(basic_three_d_kmeans_router.router, prefix="/basic", tags=["3D kmeans"])
app.include_router(advanced_three_d_kmeans_router.router, prefix="/advanced", tags=["3D kmeans"])


# ----------------------- nD kmeans Routers ---------------------------------------------
app.include_router(basic_n_d_kmeans_router.router, prefix="/basic", tags=["multi D kmeans"])
app.include_router(advanced_n_d_kmeans_router.router, prefix="/advanced", tags=["multi D kmeans"])


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
