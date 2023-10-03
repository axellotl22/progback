from typing import List
from math import sqrt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from app.models.elbow_model import ElbowVisualizationResult, Point, AxisLabel
from .utils import load_dataframe, clean_dataframe

def find_elbow_point(ks: List[int], distortions: List[float]) -> int:
    """Find the elbow point in the KMeans distortion curve."""
    max_delta = 0
    optimal_k = 1
    for i in range(1, len(distortions)):
        delta = distortions[i-1] - distortions[i]
        if delta > max_delta:
            max_delta = delta
            optimal_k = ks[i]
    return optimal_k

def process_file_for_elbow_method(file_path: str) -> ElbowVisualizationResult:
    """Process the user-provided file for the Elbow method and return visualization-friendly results."""
    data = load_dataframe(file_path)
    data = clean_dataframe(data)
    
    max_k = int(sqrt(len(data)))
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = data.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    transformers = [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
    processed_data = ct.fit_transform(data)
    
    ks = list(range(1, max_k + 1))
    average_distortions = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(processed_data)
        average_distortions.append(kmeans.inertia_)

    recommended_k = find_elbow_point(ks, average_distortions)

    points = [Point(x=k, y=distortion) for k, distortion in zip(ks, average_distortions)]
    labels = AxisLabel(x="Number of Clusters (k)", y="Average Distortion")
    recommended_point = Point(x=recommended_k, y=average_distortions[ks.index(recommended_k)])

    return ElbowVisualizationResult(points=points, labels=labels, recommended_point=recommended_point)

def process_file_for_elbow_method_optimized(file_path: str) -> ElbowVisualizationResult:
    """Optimized version using MiniBatchKMeans and PCA for visualization-friendly results."""
    data = load_dataframe(file_path)
    data = clean_dataframe(data)

    max_k = int(sqrt(len(data)))
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = data.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    transformers = [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
    processed_data = ct.fit_transform(data)

    pca = PCA(n_components=0.95)
    reduced_data = pca.fit_transform(processed_data)

    ks = list(range(1, max_k + 1))
    average_distortions = []
    for k in ks:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_data)
        average_distortions.append(kmeans.inertia_)

    recommended_k = find_elbow_point(ks, average_distortions)

    points = [Point(x=k, y=distortion) for k, distortion in zip(ks, average_distortions)]
    labels = AxisLabel(x="Number of Clusters (k)", y="Average Distortion")
    recommended_point = Point(x=recommended_k, y=average_distortions[ks.index(recommended_k)])

    return ElbowVisualizationResult(points=points, labels=labels, recommended_point=recommended_point)
