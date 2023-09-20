""" Tests für die App. """

import os
import shutil
from fastapi.testclient import TestClient
from app.main import app

# Setzen der Umgebungsvariable für den Testmodus
os.environ["TEST_MODE"] = "True"

client = TestClient(app)
ENDPOINT_URL = "/clustering/perform-kmeans-clustering/"
BASE_TEST_DIR = "test/"
TEMP_FILES_DIR = "temp_files/"

BASIC_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_basic_test.csv")
ADVANCED_TEST_FILE = os.path.join(BASE_TEST_DIR, "kmeans_advanced_test.csv")


class TestApp:
    """Testklasse für die Anwendung."""

    @classmethod
    def setup_class(cls):
        """ Überprüfen, ob die Testdateien vorhanden sind"""
        assert os.path.exists(
            BASIC_TEST_FILE), f"Datei {BASIC_TEST_FILE} nicht gefunden."
        assert os.path.exists(
            ADVANCED_TEST_FILE), f"Datei {ADVANCED_TEST_FILE} nicht gefunden."
        # Erstellen des temporären Verzeichnisses, wenn es nicht existiert
        if not os.path.exists(TEMP_FILES_DIR):
            os.makedirs(TEMP_FILES_DIR)

    @classmethod
    def teardown_class(cls):
        """ Entfernen von temporären Dateien nach Abschluss der Tests"""
        shutil.rmtree(TEMP_FILES_DIR, ignore_errors=True)

    def test_basic(self):
        """Teste den Endpoint mit allen Parametern."""
        with open(BASIC_TEST_FILE, "rb") as f:
            response = client.post(
                ENDPOINT_URL,
                files={"file": f}
            )
        assert response.status_code == 200
        assert "name" in response.json()
        assert "cluster" in response.json()

    def test_advanced(self):
        """Teste den Endpoint mit verschiedenen Kombinationen von Parametern."""
        with open(ADVANCED_TEST_FILE, "rb") as f:
            # Ohne Clusterangabe aber mit Distanzangabe
            response_1 = client.post(
                ENDPOINT_URL,
                files={"file": f},
                data={"column1": 4, 
                      "column2": 5, 
                      "distanceMetric": "JACCARDS"}
            )
            assert response_1.status_code == 200
            assert "name" in response_1.json()
            assert "cluster" in response_1.json()

            print(response_1.json())
