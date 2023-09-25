"""Tests for the app."""

import os
import shutil

from fastapi.testclient import TestClient
from app.main import app  

# Set environment variable for test mode
os.environ["TEST_MODE"] = "True"  

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
        
        assert os.path.exists(BASIC_TEST_FILE), f"File {BASIC_TEST_FILE} not found."
        assert os.path.exists(ADVANCED_TEST_FILE), f"File {ADVANCED_TEST_FILE} not found."
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(TEMP_FILES_DIR):
            os.makedirs(TEMP_FILES_DIR)

    @classmethod
    def teardown_class(cls):
        """Remove temporary files after tests complete."""
        
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

            # ToDo: More Tests

        # Print last response
        print(response_1.json())