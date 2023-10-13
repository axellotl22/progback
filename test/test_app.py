"""Tests for the app."""
import asyncio
import os
import shutil

from fastapi.testclient import TestClient

from app.main import app
from app.database import user_db, job_db

# Set environment variable for test mode
os.environ["TEST_MODE"] = "True"
os.environ["DEV_MODE"] = "False"

client = TestClient(app)

ENDPOINT_URL = "/clustering/perform-kmeans-clustering/"
BASE_TEST_DIR = "test/"
TEMP_FILES_DIR = "temp_files/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")
ADVANCED_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_advanced_test.csv")


class TestApp:
    """Test class for the application."""

    @classmethod
    def setup_class(cls):
        """Check if test files are present."""

        assert os.path.exists(
            BASIC_TEST_FILE), f"File {BASIC_TEST_FILE} not found."
        assert os.path.exists(
            ADVANCED_TEST_FILE), f"File {ADVANCED_TEST_FILE} not found."

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
        """Test endpoint with all default parameters."""

        # Load test data
        with open(BASIC_TEST_FILE, "rb") as file:
            # Make request
            response = client.post(ENDPOINT_URL, files={"file": file})

            # Validate response
            assert response.status_code == 200
            assert "name" in response.json()
            assert "cluster" in response.json()

    def test_advanced(self):
        """Test endpoint with different combinations of parameters."""

        # Load test data
        with open(ADVANCED_TEST_FILE, "rb") as file:
            # Test with no kCluster but custom distance metric
            response_1 = client.post(
                ENDPOINT_URL,
                files={"file": file},
                data={"column1": 4,
                      "column2": 5,
                      "distanceMetric": "JACCARDS"}
            )
            assert response_1.status_code == 200
            assert "name" in response_1.json()
            assert "cluster" in response_1.json()

        # Print last response
        print(response_1.json())

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
