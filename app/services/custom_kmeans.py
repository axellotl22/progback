# pylint: disable=too-few-public-methods
"""
optim_kmeans.py
---------------
A module containing optimized implementations of K-Means clustering algorithms.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numba import jit


@jit(nopython=True)
def euclidean_distance_matrix(matrix, centers):
    """
    Calculate the Euclidean distance matrix between data points and cluster centers.

    Args:
    - matrix (numpy.ndarray): Data points.
    - centers (numpy.ndarray): Cluster centers.

    Returns:
    - numpy.ndarray: Euclidean distance matrix.
    """
    diff = matrix[:, np.newaxis] - centers
    return np.sqrt((diff ** 2).sum(axis=2))


@jit(nopython=True)
def manhattan_distance_matrix(matrix, centers):
    """
    Calculate the Manhattan distance matrix between data points and cluster centers.

    Args:
    - matrix (numpy.ndarray): Data points.
    - centers (numpy.ndarray): Cluster centers.

    Returns:
    - numpy.ndarray: Manhattan distance matrix.
    """
    return np.sum(np.abs(matrix[:, np.newaxis] - centers), axis=2)


@jit(nopython=True)
def jaccard_distance_matrix(matrix, centers):
    """
    Calculate the Jaccard distance matrix between data points and cluster centers.

    Args:
    - matrix (numpy.ndarray): Data points.
    - centers (numpy.ndarray): Cluster centers.

    Returns:
    - numpy.ndarray: Jaccard distance matrix.
    """
    intersection = np.minimum(matrix[:, np.newaxis], centers).sum(axis=2)
    union = np.maximum(matrix[:, np.newaxis], centers).sum(axis=2)
    return 1 - (intersection / union)


class BaseOptimizedKMeans:
    """
    Base class for optimized K-Means clustering with specified distance metrics.

    Attributes:
    - n_clusters (int): Number of clusters.
    - distance_metric (str): Distance metric for clustering.
    - max_iterations (int): Max iterations.
    - tol (float): Tolerance.
    - iterations_ (int): Iterations during fitting.
    - cluster_centers_ (numpy.ndarray): Fitted cluster centers.
    - supported_distance_metrics (dict): Supported metrics.

    Methods:
    - __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iterations=300, tol=1e-4)
    - _assign_labels(self, data_points)
    - _handle_empty_clusters(self, data_points, labels)
    - _initialize_centers(self, data_points)
    - fit(self, data_points, n_components=None)
    """
    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance_matrix,
        "MANHATTAN": manhattan_distance_matrix,
        "JACCARDS": jaccard_distance_matrix
    }

    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iterations=300, tol=1e-4):
        """
        Initialize the KMeans clustering model.

        Args:
        - n_clusters (int): Number of clusters.
        - distance_metric (str): Distance metric for clustering.
        - max_iterations (int): Maximum number of iterations.
        - tol (float): Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.max_iterations = max_iterations
        self.tol = tol
        self.iterations_ = 0

        self.distance_func = self.supported_distance_metrics.get(
            distance_metric)
        if not self.distance_func:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        self.cluster_centers_ = None

    def assign_labels(self, data_points):
        """
        Assign data points to cluster labels.

        Args:
        - data_points (numpy.ndarray): Data points.

        Returns:
        - numpy.ndarray: Cluster labels.
        """
        distances = self.distance_func(data_points, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        self._handle_empty_clusters(data_points, labels)
        return labels

    def _handle_empty_clusters(self, data_points, labels):
        """
        Handle empty clusters by reassigning data points to them.

        Args:
        - data_points (numpy.ndarray): Data points.
        - labels (numpy.ndarray): Cluster labels.
        """
        distances_to_centers = self.distance_func(
            data_points, self.cluster_centers_)
        for i in range(self.n_clusters):
            if i not in labels:
                farthest_point_index = np.argmax(
                    np.min(distances_to_centers, axis=1))
                self.cluster_centers_[i] = data_points[farthest_point_index]
                distances_to_centers[farthest_point_index] = -np.inf
                labels[farthest_point_index] = i

    def _initialize_centers(self, data_points):
        """
        Initialize cluster centers using K-Means++.

        Args:
        - data_points (numpy.ndarray): Data points.
        """
        kmeans = KMeans(n_clusters=self.n_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

    def fit(self, data_points, n_components=None):
        """
        Fit the KMeans model to the data.

        Args:
        - data_points (numpy.ndarray): Data points.
        - n_components (int): Number of PCA components (optional).
        """
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Apply PCA for dimensionality reduction
        n_components = n_components or min(data_points.shape)
        pca = PCA(n_components=n_components)
        data_points = pca.fit_transform(data_points)

        self._initialize_centers(data_points)

        for _ in range(self.max_iterations):
            labels = self.assign_labels(data_points)
            new_centers = np.array([data_points[labels == i].mean(
                axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers
            self.iterations_ += 1


class OptimizedKMeans(BaseOptimizedKMeans):
    """
    Optimized K-Means clustering with specified distance metric.

    Attributes:
    - n_clusters (int): Number of clusters.
    - distance_metric (str): Distance metric.
    - max_iterations (int): Max iterations.
    - tol (float): Tolerance.
    - iterations_ (int): Iterations during fitting.
    - cluster_centers_ (numpy.ndarray): Fitted cluster centers.
    - supported_distance_metrics (dict): Supported metrics.

    Methods:
    - __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iterations=300, tol=1e-4)
    - _assign_labels(self, data_points)
    - _handle_empty_clusters(self, data_points, labels)
    - _initialize_centers(self, data_points)
    - fit(self, data_points, n_components=None)
    """


class OptimizedMiniBatchKMeans(BaseOptimizedKMeans):
    """
    Optimized Mini-Batch K-Means clustering with specified distance metric.

    Attributes:
    - n_clusters (int): Number of clusters.
    - distance_metric (str): Distance metric.
    - batch_size (int): Batch size for Mini-Batch KMeans.
    - max_iterations (int): Max iterations.
    - tol (float): Tolerance.
    - iterations_ (int): Iterations during fitting.
    - cluster_centers_ (numpy.ndarray): Fitted cluster centers.
    - supported_distance_metrics (dict): Supported metrics.

    Methods:
    - __init__(self, n_clusters, distance_metric="EUCLIDEAN", 
                        batch_size=100, max_iterations=300, tol=1e-4)
    - fit(self, data_points, n_components=None)
    """
    # pylint: disable=too-many-arguments
    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", 
                 batch_size=100, max_iterations=300, tol=1e-4):
        """
        Initialize the OptimizedMiniBatchKMeans model.

        Args:
        - n_clusters (int): Number of clusters.
        - distance_metric (str): Distance metric for clustering.
        - batch_size (int): Batch size for Mini-Batch KMeans.
        - max_iterations (int): Maximum number of iterations.
        - tol (float): Tolerance for convergence.
        """
        super().__init__(n_clusters, distance_metric, max_iterations, tol)
        self.batch_size = batch_size

    def fit(self, data_points, n_components=None):
        """
        Fit the OptimizedMiniBatchKMeans model to the data.

        Args:
        - data_points (numpy.ndarray): Data points.
        - n_components (int): Number of PCA components (optional).
        """
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Apply PCA for dimensionality reduction
        n_components = n_components or min(data_points.shape)
        pca = PCA(n_components=n_components)
        data_points = pca.fit_transform(data_points)

        self._initialize_centers(data_points)

        for _ in range(self.max_iterations):
            indices = np.random.choice(
                data_points.shape[0], self.batch_size, replace=False)
            mini_batch = data_points[indices]
            labels = self.assign_labels(mini_batch)
            new_centers = np.array([mini_batch[labels == i].mean(
                axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers
            self.iterations_ += 1