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
    Compute the Euclidean distance matrix between given data points and cluster centers.

    Parameters:
    - matrix (numpy.ndarray): Array representing data points.
    - centers (numpy.ndarray): Array representing cluster centers.

    Returns:
    - numpy.ndarray: The Euclidean distance matrix.
    """
    diff = matrix[:, np.newaxis] - centers
    return np.sqrt((diff ** 2).sum(axis=2))


@jit(nopython=True)
def manhattan_distance_matrix(matrix, centers):
    """
    Compute the Manhattan distance matrix between given data points and cluster centers.

    Parameters:
    - matrix (numpy.ndarray): Array representing data points.
    - centers (numpy.ndarray): Array representing cluster centers.

    Returns:
    - numpy.ndarray: The Manhattan distance matrix.
    """
    return np.sum(np.abs(matrix[:, np.newaxis] - centers), axis=2)


@jit(nopython=True)
def jaccard_distance_matrix(matrix, centers):
    """
    Compute the Jaccard distance matrix between given data points and cluster centers.

    Parameters:
    - matrix (numpy.ndarray): Array representing data points.
    - centers (numpy.ndarray): Array representing cluster centers.

    Returns:
    - numpy.ndarray: The Jaccard distance matrix.
    """
    intersection = np.minimum(matrix[:, np.newaxis], centers).sum(axis=2)
    union = np.maximum(matrix[:, np.newaxis], centers).sum(axis=2)
    return 1 - (intersection / union)


class BaseOptimizedKMeans:
    """
    Base class for optimized K-Means clustering using specified distance metrics.
    """

    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance_matrix,
        "MANHATTAN": manhattan_distance_matrix,
        "JACCARDS": jaccard_distance_matrix
    }

    def __init__(self, number_clusters, distance_metric="EUCLIDEAN",
                 max_iterations=300, tolerance=1e-4):
        """
        Constructor for the optimized K-Means clustering model.

        Parameters:
        - number_clusters (int): The desired number of clusters to form.
        - distance_metric (str, optional): Specifies the distance metric. Options are 
            "EUCLIDEAN", "MANHATTAN", or "JACCARDS". Defaults to "EUCLIDEAN".
        - max_iterations (int, optional): The maximum number of iterations for the algorithm 
            to converge. Defaults to 300.
        - tolerance (float, optional): The threshold to determine convergence. If the change 
            in cluster centers is below this value, the algorithm is considered to have converged. 
            Defaults to 1e-4.

        Raises:
        - ValueError: If an unsupported distance metric is provided.
        """
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
        Compute K-Means clustering and establish cluster centers.

        This method initializes the cluster centers using the KMeans++ algorithm, 
        iteratively refines the cluster assignments, and updates the cluster centers 
        until convergence or until reaching the maximum number of iterations.

        Parameters:
        - data_points (numpy.ndarray): The input data array where each row represents 
            an observation and each column represents a feature.

        Logs:
        - Initialization details, progress of iterations, 
                convergence status, and potential anomalies.
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
        Assign labels to observations based on their proximity to cluster centers.

        This method calculates the distance of each observation to each cluster center 
        and assigns each observation to the nearest cluster.

        Parameters:
        - data_points (numpy.ndarray): Array of observations to be labeled.

        Returns:
        - numpy.ndarray: Array containing the cluster label for each observation.

        Logs:
        - Information about data shape, computed distances, and any exceptions encountered.
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
    Optimized K-Means clustering leveraging a specified distance metric.
    Inherits from the BaseOptimizedKMeans class.
    """


class OptimizedMiniBatchKMeans(BaseOptimizedKMeans):
    """
    Optimized Mini-Batch K-Means clustering with a specified distance metric.
    
    The Mini-Batch K-Means algorithm iteratively refines cluster assignments using a 
    subset of the data called a mini-batch, improving computation time especially on large datasets.

    Attributes:
        batch_size (int): The number of data points to include in each mini-batch.
    """
    # pylint: disable=too-many-arguments

    def __init__(self, number_clusters, distance_metric="EUCLIDEAN",
                 batch_size=100, max_iterations=300, tolerance=1e-4):
        """
        Constructor for the optimized Mini-Batch K-Means clustering model.

        Parameters:
        - number_clusters (int): The desired number of clusters to form.
        - distance_metric (str, optional): Specifies the distance metric. Options are 
            "EUCLIDEAN", "MANHATTAN", or "JACCARDS". Defaults to "EUCLIDEAN".
        - batch_size (int, optional): The number of data points to include in each mini-batch.
            Defaults to 100.
        - max_iterations (int, optional): The maximum number of iterations for the algorithm 
            to converge. Defaults to 300.
        - tolerance (float, optional): The threshold to determine convergence. If the change 
            in cluster centers is below this value, the algorithm is considered to have converged. 
            Defaults to 1e-4.
        """
        super().__init__(number_clusters, distance_metric, max_iterations, tolerance)
        self.batch_size = batch_size

    def fit(self, data_points):
        """
        Compute Mini-Batch K-Means clustering and establish cluster centers.

        This method initializes the cluster centers using the KMeans++ algorithm, 
        then iteratively refines cluster assignments using mini-batches until 
        convergence or until reaching the maximum number of iterations.

        Parameters:
        - data_points (numpy.ndarray): The input data array where each row represents 
            an observation and each column represents a feature.
        """
        kmeans = KMeans(n_clusters=self.number_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iterations):
            indices = np.random.choice(
                data_points.shape[0], self.batch_size, replace=False)
            mini_batch = data_points[indices]

            distances = self.distance(mini_batch, self.cluster_centers_)
            labels = np.argmin(distances, axis=1)

            for i in range(self.number_clusters):
                points_in_cluster = mini_batch[labels == i]
                if len(points_in_cluster) > 0:
                    # Use a simple moving average for updating
                    self.cluster_centers_[i] = (0.9 * self.cluster_centers_[i]
                                                + 0.1 * points_in_cluster.mean(axis=0))

            # Check for convergence
            if np.all(np.abs(self.distance(self.cluster_centers_, 
                                           self.cluster_centers_.copy()) < self.tolerance)):
                logger.info(
                    "Convergence reached after %d iterations.", _)
                break

            self.iterations_ += 1
