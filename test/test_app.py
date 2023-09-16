""" Tests für die App. """

import os
import shutil
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering, select_columns
)

# Setzen des Test Mode auf True
os.environ["TEST_MODE"] = "True"

client = TestClient(app)
TEST_DATA_ORIGINAL_PATH = "test/test_daten.xlsx"
TEST_DATA_COPY_PATH = f"temp_files/{os.path.basename(TEST_DATA_ORIGINAL_PATH)}"

class TestApp:
    """Testklasse für die Anwendung."""

    @classmethod
    def setup_class(cls):
        """ Setup für die gesamte Testklasse. Wird einmalig vor allen Tests ausgeführt. """
        assert os.path.exists(TEST_DATA_ORIGINAL_PATH), f"Quelldatei {TEST_DATA_ORIGINAL_PATH} nicht gefunden."
        if not os.path.exists("temp_files/"):
            os.makedirs("temp_files/")
        shutil.copy(TEST_DATA_ORIGINAL_PATH, TEST_DATA_COPY_PATH)

    @classmethod
    def teardown_class(cls):
        """ Teardown für die gesamte Testklasse. Wird einmalig nach allen Tests ausgeführt. """
        if os.path.exists(TEST_DATA_COPY_PATH):
            os.remove(TEST_DATA_COPY_PATH)

    def test_setup(self):
        """ Überprüft die Initialisierung des API-Endpunkts und der Dienste. """
        assert app is not None
        assert load_dataframe is not None
        assert clean_dataframe is not None
        assert determine_optimal_clusters is not None
        assert perform_clustering is not None

    def test_clustering_endpoint_with_specified_clusters(self):
        """Testet den Clustering-Endpunkt mit einer vom Benutzer angegebenen Clusteranzahl."""
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post("/clustering/perform-kmeans-clustering/", files={"file": file}, data={"clusters": 3})
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "centroids" in data
        assert "point_to_centroid_mappings" in data
        assert "x_label" in data
        assert "y_label" in data
        actual_clusters = len(data["centroids"])
        assert actual_clusters >= 2

    def test_clustering_endpoint(self):
        """Testet den Clustering-Endpunkt."""
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post("/clustering/perform-kmeans-clustering/", files={"file": file})
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "centroids" in data
        assert "point_to_centroid_mappings" in data
        assert "x_label" in data
        assert "y_label" in data
        data_frame = pd.read_excel(TEST_DATA_COPY_PATH)
        assert len(data["points"]) == len(data_frame)

    def test_clustering_endpoint_with_columns_as_string(self):
        """Testet den Clustering-Endpunkt mit columns als String-Form."""
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post("/clustering/perform-kmeans-clustering/", files={"file": file}, data={"columns": "[0,2]"})
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "centroids" in data
        assert "point_to_centroid_mappings" in data
        assert "x_label" in data
        assert "y_label" in data

    def test_clustering_endpoint_with_columns_as_list(self):
        """Testet den Clustering-Endpunkt mit columns als Listenform."""
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post("/clustering/perform-kmeans-clustering/", files={"file": file}, data={"columns": "0,2"})
        assert response.status_code == 200
        data = response.json()
        assert "points" in data
        assert "centroids" in data
        assert "point_to_centroid_mappings" in data
        assert "x_label" in data
        assert "y_label" in data

    def test_dataframe_functions(self):
        """Testet die DataFrame-Hilfsfunktionen mit den Testdaten."""
        data_frame = load_dataframe(TEST_DATA_COPY_PATH)
        assert isinstance(data_frame, pd.DataFrame)
        assert not data_frame.empty
        
        cleaned_data_frame = clean_dataframe(data_frame)
        assert not cleaned_data_frame.empty
        
        optimal_clusters = determine_optimal_clusters(cleaned_data_frame)
        assert optimal_clusters > 0
        
        kmeans_results = perform_clustering(cleaned_data_frame, optimal_clusters)
        
        # Extrahieren der Cluster-Zuordnungen aus den 'points'
        labels = [point['cluster'] for point in kmeans_results['points']]
        assert len(labels) == len(cleaned_data_frame)


    def test_select_columns_with_test_data(self):
        """Testet die select_columns-Hilfsfunktion mit den Testdaten."""
        data_frame = load_dataframe(TEST_DATA_COPY_PATH)
        selected_df = select_columns(data_frame, columns=[0, 2])
        assert list(selected_df.columns) == ["Merkmal 1", "Merkmal 3"]
        assert selected_df.shape[1] == 2
