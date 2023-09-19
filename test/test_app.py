""" Tests für die App. """

import os
import shutil
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

from app.services.clustering_service import (
    determine_optimal_clusters, perform_clustering
)
from app.services.utils_service import (
    load_dataframe, clean_dataframe, select_columns
)

os.environ["TEST_MODE"] = "True"

client = TestClient(app)
TEST_DATA_ORIGINAL_PATH = "test/test_daten.xlsx"
TEST_DATA_COPY_PATH = f"temp_files/{os.path.basename(TEST_DATA_ORIGINAL_PATH)}"


class TestApp:
    """Testklasse für die Anwendung."""

    @classmethod
    def setup_class(cls):
        assert os.path.exists(
            TEST_DATA_ORIGINAL_PATH), f"Quelldatei {TEST_DATA_ORIGINAL_PATH} nicht gefunden."
        if not os.path.exists("temp_files/"):
            os.makedirs("temp_files/")
        shutil.copy(TEST_DATA_ORIGINAL_PATH, TEST_DATA_COPY_PATH)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(TEST_DATA_COPY_PATH):
            os.remove(TEST_DATA_COPY_PATH)

    def test_setup(self):
        assert app is not None
        assert load_dataframe is not None
        assert clean_dataframe is not None
        assert determine_optimal_clusters is not None
        assert perform_clustering is not None

    def test_clustering_endpoint_with_specified_clusters(self):
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post(
                "/clustering/perform-kmeans-clustering/",
                files={"file": file}, data={"clusters": 3}
            )
        assert response.status_code == 200
        data = response.json()
        assert "cluster" in data
        assert "name" in data

    def test_clustering_endpoint(self):
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post(
                "/clustering/perform-kmeans-clustering/", files={"file": file})
        assert response.status_code == 200
        data = response.json()
        assert "cluster" in data
        assert "name" in data

    def test_clustering_endpoint_with_column_inputs(self):
        with open(TEST_DATA_COPY_PATH, "rb") as file:
            response = client.post("/clustering/perform-kmeans-clustering/",
                                   files={"file": file}, data={"columns": "[0,2]"})
        assert response.status_code == 200
        data = response.json()
        assert "cluster" in data
        assert "name" in data

    def test_dataframe_functions(self):
        data_frame = load_dataframe(TEST_DATA_COPY_PATH)
        assert isinstance(data_frame, pd.DataFrame)
        assert not data_frame.empty

        cleaned_data_frame = clean_dataframe(data_frame)
        assert not cleaned_data_frame.empty

        optimal_clusters = determine_optimal_clusters(cleaned_data_frame)
        assert optimal_clusters > 0

        kmeans_results = perform_clustering(
            cleaned_data_frame, optimal_clusters)

        labels = [cluster['clusterNr']
                  for cluster in kmeans_results['cluster'] for _ in cluster['points']]
        assert len(labels) == len(cleaned_data_frame)

    def test_select_columns_with_test_data(self):
        data_frame = load_dataframe(TEST_DATA_COPY_PATH)
        selected_df = select_columns(data_frame, columns=[0, 2])
        assert list(selected_df.columns) == ["Merkmal 1", "Merkmal 3"]
        assert selected_df.shape[1] == 2
