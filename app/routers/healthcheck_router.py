"""
healtchecks_router.py
---------------

Provides Enpoint for Healtchcheck
"""
import os

from urllib import request, parse

from fastapi import APIRouter
from fastapi.testclient import TestClient

router = APIRouter()


ENDPOINT_URL = "/clustering/perform-kmeans-clustering/"
BASE_TEST_DIR = "test/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")


@router.get("/health")
async def healtcheck():
    try:


        with open(BASIC_TEST_FILE, "rb") as file:

            # Make request
            response = client.post(ENDPOINT_URL, files={"file": file})
    except Exception:
        return "", 500

    return "", 200
