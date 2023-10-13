"""Tests for the app."""
import random
import asyncio
import os
import shutil
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.database import user_db, job_db

# Set environment variables for test mode
os.environ["TEST_MODE"] = "True"
os.environ["DEV_MODE"] = "False"

client = TestClient(app)

BASE_TEST_DIR = "test/"
TEMP_FILES_DIR = "temp_files/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")
ADVANCED_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_advanced_test.csv")

ENDPOINTS = {
    "2d": ["/basic/perform-2d-kmeans/", "/advanced/perform-advanced-2d-kmeans/"],
    "3d": ["/basic/perform-3d-kmeans/", "/advanced/perform-advanced-3d-kmeans/"],
    "nd": ["/basic/perform-nd-kmeans/", "/advanced/perform-advanced-nd-kmeans/"],
}

ELBOW_ENDPOINT = "/determination/elbow"

class TestApp:
    """Test class for the application."""

    @classmethod
    def setup_class(cls):
        """Check if test files are present."""
        assert os.path.exists(BASIC_TEST_FILE), f"File {BASIC_TEST_FILE} not found."
        assert os.path.exists(ADVANCED_TEST_FILE), f"File {ADVANCED_TEST_FILE} not found."

        # Create temp directory if it doesn't exist
        if not os.path.exists(TEMP_FILES_DIR):
            os.makedirs(TEMP_FILES_DIR)

        # Setup sql tables
        asyncio.run(user_db.create_db_and_tables())
        asyncio.run(job_db.create_db_and_tables())

    @classmethod
    def teardown_class(cls):
        """Remove temporary files after tests complete."""
        os.remove("test/test.db")
        shutil.rmtree(TEMP_FILES_DIR, ignore_errors=True)

    def test_basic(self):
        """Test basic endpoints."""
        for cluster_type, (basic_endpoint, _) in ENDPOINTS.items():
            with open(BASIC_TEST_FILE, "rb") as file:
                data = {
                    "k_clusters": random.choice([3, 4, 5, 6, 7]),
                }

                if cluster_type == "3d":
                    data.update({"column3": 6})

                response = client.post(basic_endpoint, files={"file": file}, data=data)

                assert response.status_code == 200, f"Expected 200 status for {basic_endpoint}, got {response.status_code}"
                assert "name" in response.json(), f"Expected 'name' key in response for {basic_endpoint}"
                assert "cluster" in response.json(), f"Expected 'cluster' key in response for {basic_endpoint}"

    def test_advanced(self):
        """Test advanced endpoints."""
        for cluster_type, (_, advanced_endpoint) in ENDPOINTS.items():
            with open(ADVANCED_TEST_FILE, "rb") as file:
                data = {
                    "column1": 4,
                    "column2": 5,
                    "distanceMetric": "JACCARDS"
                }

                if cluster_type == "3d":
                    data.update({"column3": 6})
                elif cluster_type == "nd":
                    data.update({"use_3d_model": random.choice([True, False])})

                response = client.post(advanced_endpoint, files={"file": file}, data=data)

                assert response.status_code == 200
                assert "name" in response.json()
                assert "cluster" in response.json()

    def test_elbow(self):
        """Test the elbow determination endpoint."""
        with open(BASIC_TEST_FILE, "rb") as file:
            response = client.post(ELBOW_ENDPOINT, files={"file": file})

            assert response.status_code == 200

    def test_jobs(self):
        """
        Tests the job functionality
        """
        # Create an account
        reg_json = {
            "email": "test@test.com",
            "password": "test",
            "username": "test"
        }
        client.post("/register/", json=reg_json)

        # Login
        login_json = {
            "username": "test@test.com",
            "password": "test",
        }
        client.post("/login/", data=login_json)
        auth_cookie = client.cookies.get("fastapiusersauth")

        assert auth_cookie is not None

        # Create a job
        job_json = {
            "name": "Test-Basic-2d-KMeans",
            "column1": 0,
            "column2": 1,
            "distance_metric": "EUCLIDEAN",
            "kmeans_type": "OptimizedKMeans",
            "k_clusters": 3,
            "normalize": True
        }
        with open(BASIC_TEST_FILE, "rb") as file:
            result = client.post("/jobs/create/basic_2d_kmeans",
                                 data=job_json, files={"file": file},
                                 headers={"Cookie": "fastapiusersauth=" + auth_cookie})

        job_id = result.json()["job_id"]
        assert job_id is not None

        # Connect to websocket and run job
        with (client.websocket_connect(f"/jobs/{job_id}/",
                                      headers={"Cookie": "fastapiusersauth=" + auth_cookie})
              as socket):
            data = socket.receive_json()

            assert "name" in data
            assert "cluster" in data

@pytest.mark.asyncio
async def test_db_initialization():
    """Test database initialization."""
    await user_db.create_db_and_tables()
    await job_db.create_db_and_tables()
    assert True
