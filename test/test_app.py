""" Tests für die App. """

import os
import pandas as pd

from fastapi.testclient import TestClient

from app.main import app
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering
)

client = TestClient(app)

TEST_DATA_PATH = "test/test_daten.xlsx"


class TestApp:
    """Test-Klasse für die Anwendung."""

    def test_setup(self):
        """ Tests die Einrichtung des API-Endpunkts und der Dienste. """

        assert app is not None
        assert load_dataframe is not None
        assert clean_dataframe is not None
        assert determine_optimal_clusters is not None
        assert perform_clustering is not None

    def test_clustering_endpoint(self):
        
        """ 
        Testet den Clustering-Endpunkt. 
        Hochladen einer Testdatei und Überprüfung der Antwort. 
        """

        with open(TEST_DATA_PATH, "rb") as file:
            response = client.post("/clustering/upload/", files={"file": file})

        assert response.status_code == 200

        data = response.json()
        assert "cluster_labels" in data
        assert "optimal_cluster_count" in data

        uploaded_file_path = f"temp_files/{os.path.basename(TEST_DATA_PATH)}"
        data_frame = pd.read_excel(uploaded_file_path)

        assert len(data["cluster_labels"]) == len(data_frame)
        assert data["optimal_cluster_count"] > 0

    def test_dataframe_functions(self):
        """ Testet die DataFrame-Hilfsfunktionen mit den Testdaten. """

        data_frame = load_dataframe(TEST_DATA_PATH)
        assert isinstance(data_frame, pd.DataFrame)
        assert not data_frame.empty

        cleaned_data_frame = clean_dataframe(data_frame)
        assert not cleaned_data_frame.empty

        optimal_clusters = determine_optimal_clusters(cleaned_data_frame)
        assert optimal_clusters > 0

        labels = perform_clustering(cleaned_data_frame, optimal_clusters)
        assert len(labels) == len(cleaned_data_frame)
