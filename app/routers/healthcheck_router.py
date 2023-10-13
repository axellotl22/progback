"""
healthchecks_router.py
----------------------

Provides an endpoint for health checks. This router is used to periodically check
the health of the various endpoints related to k-means clustering. It posts sample data
to these endpoints and expects a successful response.
"""

import os
import random
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Base configurations
BASE_URL = 'http://localhost:8080'
BASE_TEST_DIR = "test/"
TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_advanced_test.csv")

# Lists of possible configurations for k-means clustering
DISTANCE_METRICS = ["EUCLIDEAN", "JACCARDS", "MANHATTAN", "COSINE"]
KMEANS_TYPES = ["OptimizedKMeans", "OptimizedMiniBatchKMeans"]
METHODS = ["standard", "optimized"]

# Endpoint mappings for different dimensional k-means clustering
ENDPOINTS = {
    "2d": ["/basic/perform-2d-kmeans/", "/advanced/perform-advanced-2d-kmeans/"],
    "3d": ["/basic/perform-3d-kmeans/", "/advanced/perform-advanced-3d-kmeans/"],
    "nd": ["/basic/perform-nd-kmeans/", "/advanced/perform-advanced-nd-kmeans/"],
}


@router.get("/health")
async def healthcheck():
    """
    Endpoint for health checks.
    Iterates over all k-means endpoints and tests them with sample data.
    """
    for cluster_type, (basic_endpoint, advanced_endpoint) in ENDPOINTS.items():
        # Basic health check
        if not await perform_check(basic_endpoint, TEST_FILE, cluster_type, advanced=False):
            return return_exception(basic_endpoint)

        # Advanced health check with varying distance metrics
        if not await perform_check(advanced_endpoint, TEST_FILE, cluster_type, advanced=True):
            return return_exception(advanced_endpoint)

    # Additional check for elbow determination method
    if not await perform_elbow_check("/determination/elbow", TEST_FILE):
        return return_exception("/determination/elbow")

    return {'detail': 'Healthcheck Successful'}


async def perform_check(endpoint_url: str, test_file: str, 
                        cluster_type: str, advanced: bool) -> bool:
    """
    Performs a health check for a given endpoint by posting sample data.

    Args:
        endpoint_url (str): URL of the endpoint to check.
        test_file (str): Path to the sample data file.
        cluster_type (str): Dimensionality of the data ('2d', '3d', or 'nd').
        advanced (bool): Whether it's a basic or advanced k-means clustering.

    Returns:
        bool: True if the endpoint responds successfully, False otherwise.
    """
    try:
        with open(test_file, "rb") as file:
            data = {
                "k_clusters": random.choice([3, 4, 5, 6, 7]),
                "kmeans_type": random.choice(KMEANS_TYPES)
            }

            # Update the data dictionary for advanced k-means clustering
            if advanced:
                data.update({
                    "column1": 4,
                    "column2": 5,
                    "distanceMetric": random.choice(DISTANCE_METRICS)
                })

            # Configuration adjustments based on data dimensionality
            if cluster_type != "nd":
                data["normalize"] = random.choice([True, False])
            else:
                data["use_3d_model"] = random.choice([True, False])

            # Post the data to the endpoint and await response
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    BASE_URL + endpoint_url,
                    files={'file': ('test.csv', file)},
                    data=data,
                    timeout=1200
                )

        return response.status_code == 200
    except Exception as error:
        # Propagate the exception as an HTTP error
        raise HTTPException(500, str(error)) from error


async def perform_elbow_check(endpoint_url: str, test_file: str) -> bool:
    """
    Specifically checks the health of the elbow determination endpoint.

    Args:
        endpoint_url (str): URL of the elbow determination endpoint.
        test_file (str): Path to the sample data file.

    Returns:
        bool: True if the endpoint responds successfully, False otherwise.
    """
    try:
        with open(test_file, "rb") as file:
            data = {
                "method": random.choice(METHODS)
            }

            # Post the data to the endpoint and await response
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    BASE_URL + endpoint_url,
                    files={'file': ('test.csv', file)},
                    data=data,
                    timeout=1200
                )

        return response.status_code == 200
    except Exception as error:
        # Propagate the exception as an HTTP error
        raise HTTPException(500, str(error)) from error


def return_exception(endpoint_url: str):
    """
    Constructs an error message for a given endpoint.

    Args:
        endpoint_url (str): URL of the failed endpoint.

    Returns:
        HTTPException: FastAPI exception indicating the failure.
    """
    return HTTPException(500, {'detail': f"Error in Endpoint {endpoint_url}"})
