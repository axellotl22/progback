"""
Tests für die App.
"""
from fastapi.testclient import TestClient
from app.main import app
from app.services.clustering_service import (
    load_dataframe, clean_dataframe, determine_optimal_clusters,
    perform_clustering
)
import pandas as pd

client = TestClient(app)

def test_setup():
    """
    Tests the setup of the API endpoint and services.
    """
    assert app is not None
    assert load_dataframe is not None
    assert clean_dataframe is not None
    assert determine_optimal_clusters is not None
    assert perform_clustering is not None

def test_clustering_endpoint():
    """
    Tests the clustering endpoint by uploading a test file and checking the response.
    """
    with open("test/test_daten.xlsx", "rb") as file:
        response = client.post("/clustering/upload/", files={"file": file})
    
    assert response.status_code == 200
    data = response.json()
    assert "cluster_labels" in data
    assert "optimal_cluster_count" in data

    # Ensure that the number of cluster labels matches the number of rows in the test data
    df = pd.read_excel("test/test_daten.xlsx")
    assert len(data["cluster_labels"]) == len(df)

    # Ensure that the optimal cluster count is a positive integer
    assert data["optimal_cluster_count"] > 0

def test_dataframe_functions():
    """
    Tests the DataFrame utility functions using the test data.
    """
    df = load_dataframe("test/test_daten.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    cleaned_df = clean_dataframe(df)
    assert not cleaned_df.empty

    optimal_clusters = determine_optimal_clusters(cleaned_df)
    assert optimal_clusters > 0

    labels = perform_clustering(cleaned_df, optimal_clusters)
    assert len(labels) == len(cleaned_df)
