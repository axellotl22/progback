import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from numba import jit

@jit(nopython=True)
def euclidean_distance_matrix(matrix, centers):
    return np.linalg.norm(matrix[:, np.newaxis] - centers, axis=2)

@jit(nopython=True)
def manhattan_distance_matrix(matrix, centers):
    return np.sum(np.abs(matrix[:, np.newaxis] - centers), axis=2)

@jit(nopython=True)
def jaccard_distance_matrix(matrix, centers):
    intersection = np.minimum(matrix[:, np.newaxis], centers).sum(axis=2)
    union = np.maximum(matrix[:, np.newaxis], centers).sum(axis=2)
    return 1 - (intersection / union)

class BaseOptimizedKMeans:
    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance_matrix,
        "MANHATTAN": manhattan_distance_matrix,
        "JACCARDS": jaccard_distance_matrix
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
        distances = self.distance_func(data_points, self.cluster_centers_)
        return np.argmin(distances, axis=1)

class OptimizedKMeans(BaseOptimizedKMeans):

    def fit(self, data_points):
        # Normalize data 
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Initial assignment using sklearn's KMeans++
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            labels = self._assign_labels(data_points)
            new_centers = np.array([data_points[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers

class OptimizedMiniBatchKMeans(BaseOptimizedKMeans):

    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", batch_size=100, max_iterations=300, tol=1e-4):
        super().__init__(n_clusters, distance_metric, max_iterations, tol)
        self.batch_size = batch_size

    def fit(self, data_points):
        # Normalize data
        scaler = StandardScaler()
        data_points = scaler.fit_transform(data_points)

        # Initial assignment using sklearn's MiniBatch KMeans
        mb_kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', batch_size=self.batch_size, n_init=10)
        mb_kmeans.fit(data_points)
        self.cluster_centers_ = mb_kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            indices = np.random.choice(data_points.shape[0], self.batch_size, replace=False)
            mini_batch = data_points[indices]
            labels = self._assign_labels(mini_batch)
            new_centers = np.array([mini_batch[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.linalg.norm(new_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_centers
