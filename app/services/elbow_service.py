"""
elbow_method_service.py
-----------------------
This module contains functions for running the KMeans Elbow method.
"""
from typing import List, Tuple
from math import sqrt

import numpy as np
import pandas as pd


from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from app.models.elbow_model import ElbowResult, DataPoint, AxisLabel
from .utils import load_dataframe, clean_dataframe


def find_elbow_point(k_s: List[int], distortions: List[float]) -> int:
    """
    Find the elbow point in the KMeans distortion curve.
    The elbow point indicates the optimal number of clusters.
    """
    max_delta = 0
    optimal_k = 1
    for i in range(1, len(distortions)):
        delta = distortions[i-1] - distortions[i]
        if delta > max_delta:
            max_delta = delta
            optimal_k = k_s[i]
    return optimal_k


def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses the input data using column transformation.
    """
    numeric_features = data.select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_features = data.select_dtypes(
        exclude=["int64", "float64"]).columns.tolist()

    transformers = [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]

    c_t = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return c_t.fit_transform(data)


def compute_kmeans(data: np.ndarray, k_s: List[int], 
                   model_class: any) -> Tuple[List[int], List[float]]:
    """
    Computes KMeans or MiniBatchKMeans and returns k values and their corresponding distortions.
    """
    average_distortions = []
    for k in k_s:
        kmeans = model_class(n_clusters=k, random_state=42)
        kmeans.fit(data)
        average_distortions.append(kmeans.inertia_)
    return k_s, average_distortions


def process_file_for_elbow_method(file_path: str, model_class: any = KMeans, 
                                  use_pca: bool = False) -> ElbowResult:
    """
    Process file and run KMeans to generate elbow curve results.
    This function can be used for both standard and optimized (with PCA) KMeans.
    """
    data = load_dataframe(file_path)
    data = clean_dataframe(data)
    max_k = int(sqrt(len(data)))
    processed_data = preprocess_data(data)

    if use_pca:
        pca = PCA(n_components=0.95)
        processed_data = pca.fit_transform(processed_data)

    k_s, average_distortions = compute_kmeans(
        processed_data, list(range(1, max_k + 1)), model_class)

    recommended_k = find_elbow_point(k_s, average_distortions)
    points = [DataPoint(x=k, y=distortion)
              for k, distortion in zip(k_s, average_distortions)]
    labels = AxisLabel(x="Number of Clusters (k)", y="Average Distortion")
    recommended_point = DataPoint(
        x=recommended_k, y=average_distortions[k_s.index(recommended_k)])

    return ElbowResult(points=points, labels=labels, recommended_point=recommended_point)


def run_standard_elbow_method(file_path: str) -> ElbowResult:
    """
    Wrapper for the standard KMeans elbow method.
    """
    return process_file_for_elbow_method(file_path)


def run_optimized_elbow_method(file_path: str) -> ElbowResult:
    """
    Wrapper for the optimized (with PCA) MiniBatchKMeans elbow method.
    """
    return process_file_for_elbow_method(file_path, model_class=MiniBatchKMeans, use_pca=True)
