"""
healtchecks_router.py
---------------

Provides Enpoint for Healtchcheck
"""
import os

import httpx

from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_URL = 'http://localhost:8080'
BASE_TEST_DIR = "test/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")
ADVANCED_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_advanced_test.csv")


@router.get("/health")
async def healtcheck():
    """
    Funktion für alle healthchecks
    """

    try:
        endpoint_url = "/clustering/perform-kmeans-clustering/"

        with open(BASIC_TEST_FILE, "rb") as file:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    BASE_URL + endpoint_url,
                    files={'file': ('test.csv', file)},
                    timeout=1200,
                )

        if response.status_code != 200:
            return return_exception(endpoint_url)

        with open(ADVANCED_TEST_FILE, "rb") as file:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    BASE_URL + endpoint_url,
                    files={'file': ('test.csv', file)},
                    data={"column1": 4,
                          "column2": 5,
                          "distanceMetric": "JACCARDS"},
                    timeout=1200,
                )

        if response.status_code != 200:
            return return_exception(endpoint_url)

    except Exception as error:
        raise HTTPException(500, str(error)) from error

    return {'detail': 'Healtcheck Successfull'}


def return_exception(endpoint_url: str):
    """ Creates Error Message """
    return HTTPException(500, {'detail': f"Error in Endpoint {endpoint_url}"})
