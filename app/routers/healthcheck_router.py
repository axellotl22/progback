"""
healtchecks_router.py
---------------

Provides Enpoint for Healtchcheck
"""
import os

import httpx
import requests
import traceback

from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_URL = os.environ['BASE_URL']

ENDPOINT_URL = "/clustering/perform-kmeans-clustering/"
BASE_TEST_DIR = "test/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")


@router.get("/health")
async def healtcheck():
    try:
        with open(BASIC_TEST_FILE, "rb") as file:
            async with httpx.AsyncClient() as client:
                response = await client.post(BASE_URL + ENDPOINT_URL)

    except Exception as e:
        raise HTTPException(500, str(e))

    return 200
