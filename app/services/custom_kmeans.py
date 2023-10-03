"""
Implementation of Optimized K-Means and MiniBatch K-Means Clustering with custom distance metrics.
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

# Distance functions
def euclidean_distance(point_a, point_b):
    """Calculates the euclidean distance between two points."""
    return np.sqrt(np.sum((point_a - point_b) ** 2))

def manhattan_distance(point_a, point_b):
    """Calculates the manhattan distance between two points."""
    return np.sum(np.abs(point_a - point_b))

def jaccard_distance(point_a, point_b):
    """Calculates the jaccard distance between two points."""
    intersection = np.minimum(point_a, point_b).sum()
    union = np.maximum(point_a, point_b).sum()
    return 1 - (intersection / union)


class BaseOptimizedKMeans:
    """Base class for optimized KMeans clustering with custom distance metrics."""

    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance,
        "MANHATTAN": manhattan_distance,
        "JACCARDS": jaccard_distance
    }

    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iterations=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.max_iterations = max_iterations
        self.tol = tol

        # Set distance function
        self.distance_func = self.supported_distance_metrics.get(distance_metric)
        if not self.distance_func:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Initial cluster centers
        self.cluster_centers_ = None

    def _assign_labels(self, data_points):
        """Assign each data point to the closest cluster center."""
        with ProcessPoolExecutor() as executor:
            distances = np.array(list(executor.map(self._compute_distances, data_points)))
        return np.argmin(distances, axis=1)

    def _compute_distances(self, point):
        """Compute distances from a point to all cluster centers."""
        return [self.distance_func(point, center) for center in self.cluster_centers_]


class OptimizedKMeans(BaseOptimizedKMeans):
    """Optimized KMeans clustering with custom distance metrics and parallel processing."""

    def fit(self, data_points):
        # Normalize data
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Initial assignment using sklearn's KMeans++
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1, max_iter=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            labels = self._assign_labels(data_points)
            new_centers = np.array([data_points[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers


class OptimizedMiniBatchKMeans(BaseOptimizedKMeans):
    """Optimized MiniBatch KMeans clustering with custom distance metrics and parallel processing."""

    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", batch_size=100, max_iterations=300, tol=1e-4):
        super().__init__(n_clusters, distance_metric, max_iterations, tol)
        self.batch_size = batch_size

    def fit(self, data_points):
        # Normalize data
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Initial assignment using sklearn's KMeans++
        mb_kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', batch_size=self.batch_size, n_init=1, max_iter=1)
        mb_kmeans.fit(data_points)
        self.cluster_centers_ = mb_kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            mini_batch = data_points[np.random.choice(data_points.shape[0], self.batch_size, replace=False)]
            labels = self._assign_labels(mini_batch)
            new_centers = np.array([mini_batch[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers

