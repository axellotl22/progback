# pylint: disable=too-few-public-methods
"""
custom_kmeans.py
---------------
A module containing optimized implementations of K-Means clustering algorithms.
"""
import logging
import numpy as np
from sklearn.cluster import KMeans
from numba import jit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


# @jit(nopython=True)
# def jaccard_distance_matrix(matrix, centers):
#    """
#    Calculate the Jaccard distance matrix between data points and cluster centers.
#
#    Args:
#    - matrix (numpy.ndarray): Data points.
#    - centers (numpy.ndarray): Cluster centers.
#
#    Returns:
#    - numpy.ndarray: Jaccard distance matrix.
#    """
#    intersection = np.minimum(matrix[:, np.newaxis], centers).sum(axis=2)
#    union = np.maximum(matrix[:, np.newaxis], centers).sum(axis=2)
#    return 1 - (intersection / union)


class BaseOptimizedKMeans:
    """
    Base class for optimized K-Means clustering with specified distance metrics.
    """

    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance_matrix,
        "MANHATTAN": manhattan_distance_matrix
        # "JACCARDS": jaccard_distance_matrix
    }

    def __init__(self, number_clusters, distance_metric="EUCLIDEAN",
                 max_iterations=300, tolerance=1e-4):
        self.number_clusters = number_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.iterations_ = 0

        self.distance = self.supported_distance_metrics.get(distance_metric)
        if self.distance is None:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        self.cluster_centers_ = None

    def fit(self, data_points):
        """
        Fit the KMeans clustering model.
        """
        logger.info("Starting fit method.")

        # Initial cluster centers using KMeans
        kmeans = KMeans(n_clusters=self.number_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_
        logger.info("Initialized cluster centers with KMeans. Cluster centers shape: %s", str(
            self.cluster_centers_.shape))

        for iteration in range(self.max_iterations):
            if np.any(np.isnan(data_points)):
                logger.error("NaN values detected in data_points!")
                return
            if np.any(np.isnan(self.cluster_centers_)):
                logger.error("NaN values detected in cluster_centers!")
                return

            logger.info("Starting loop iteration: %d of %d",
                        iteration + 1, self.max_iterations)

            logger.info("Calculating distances...")
            distances = self.distance(data_points, self.cluster_centers_)
            logger.info("Distances calculated.")

            logger.info("Assigning labels based on distances...")
            labels = np.argmin(distances, axis=1)
            logger.info("Labels assigned.")

            logger.info("Calculating new centers...")
            new_centers = np.array([data_points[labels == i].mean(axis=0)
                                    for i in range(self.number_clusters)])
            logger.info("New centers calculated.")

            if np.any(np.isnan(new_centers)):
                logger.error("NaN values detected in new centers!")
                break

            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tolerance):
                logger.info(
                    "Convergence reached after %d iterations.", iteration)
                break

            self.cluster_centers_ = new_centers
            self.iterations_ += 1
            logger.info(
                "Finished iteration %d. Updated cluster centers.", iteration)
        else:
            logger.warning("Max iterations reached. Possible non-convergence.")

    def assign_labels(self, data_points):
        """
        Assign labels to data points based on the fitted model.
        """
        logger.info("Starting assign_labels method.")

        # Log the shape of the data_points for better clarity
        logger.info("Received data_points with shape: %s",
                    str(data_points.shape))

        # Calculate distances
        try:
            distances = self.distance(data_points, self.cluster_centers_)
            logger.info("Calculated distances. Shape: %s",
                        str(distances.shape))
        except Exception as exception:
            logger.error("Error while calculating distances: %s",
                         str(exception))
            raise exception

        # Assign labels based on distances
        try:
            labels = np.argmin(distances, axis=1)
            logger.info("Assigned labels. Total labels: %d", len(labels))
        except Exception as exception:
            logger.error("Error while assigning labels: %s", str(exception))
            raise exception

        return labels


class OptimizedKMeans(BaseOptimizedKMeans):
    """
    Optimized K-Means clustering with specified distance metric.
    """


class OptimizedMiniBatchKMeans(BaseOptimizedKMeans):
    """
    Optimized Mini-Batch K-Means clustering with specified distance metric.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, number_clusters, distance_metric="EUCLIDEAN",
                 batch_size=100, max_iterations=300, tolerance=1e-4):
        super().__init__(number_clusters, distance_metric, max_iterations, tolerance)
        self.batch_size = batch_size

    def fit(self, data_points):

        kmeans = KMeans(n_clusters=self.number_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            indices = np.random.choice(
                data_points.shape[0], self.batch_size, replace=False)
            mini_batch = data_points[indices]

            distances = np.array([[self.distance(point, center)
                                   for center in self.cluster_centers_] for point in mini_batch])
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([mini_batch[labels == i].mean(axis=0)
                                    for i in range(self.number_clusters)])
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tolerance):
                break

            self.cluster_centers_ = new_centers
            self.iterations_ += 1
