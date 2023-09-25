"""
Implementation of K-Means Clustering with custom distance metrics.
"""

import numpy as np 

from sklearn.cluster import KMeans

# Distance functions

def euclidean_distance(point_a, point_b):
    """
    Calculates the euclidean distance between two points.
    """

    # Calculate squared differences between each coordinate
    squared_diffs = (point_a - point_b) ** 2
    
    # Sum the squared differences
    sum_squared_diffs = np.sum(squared_diffs) 
    
    # Take square root to get final distance
    return np.sqrt(sum_squared_diffs)

def manhattan_distance(point_a, point_b):
    """
    Calculates the manhattan distance between two points. 
    """
    
    # Calculate absolute differences between coordinates
    abs_diffs = np.abs(point_a - point_b)
    
    # Sum the absolute differences
    return np.sum(abs_diffs)

def jaccard_distance(point_a, point_b):
    """
    Calculates the jaccard distance between two points.
    """

    # Element-wise minimum to find intersection
    intersection = np.minimum(point_a, point_b)
    
    # Sum intersection to get size
    intersection_size = intersection.sum()
    
    # Element-wise maximum to find union
    union = np.maximum(point_a, point_b)
    
    # Sum union to get size
    union_size = union.sum()
    
    # Calculate jaccard distance using sizes
    return 1 - (intersection_size / union_size)

class CustomKMeans:
    """
    Implements KMeans clustering with different distance metrics.
    """

    # Dictionary to store supported distance metrics
    supported_distance_metrics = {
        "EUCLIDEAN": euclidean_distance,
        "MANHATTAN": manhattan_distance, 
        "JACCARDS": jaccard_distance
    }

    def __init__(self, number_clusters, distance_metric="EUCLIDEAN", 
                 max_iterations=300, tolerance=1e-4):
        """
        Initialization of CustomKMeans with parameters.
        """
        
        # Set attributes 
        self.number_clusters = number_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize mutable state
        self.cluster_centers_ = None 
        self.labels_ = None
        self.iterations_ = 0
        
        # Get distance function from metrics dictionary
        self.distance = self.supported_distance_metrics.get(distance_metric)
        
        # Check if distance metric is valid
        if self.distance is None:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def fit(self, data_points):
        """
        Trains the K-Means model on the provided data points.
        """

        # Initial cluster centers using KMeans
        kmeans = KMeans(n_clusters=self.number_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        
        # Set initial centers from fitted KMeans
        self.cluster_centers_ = kmeans.cluster_centers_

        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):

            # Calculate distances to cluster centers
            distances = np.array([[self.distance(point, center)
                                   for center in self.cluster_centers_] for point in data_points])
            
            # Assign points to closest cluster
            labels = np.argmin(distances, axis=1)

            # Calculate new cluster centers as means of points
            new_centers = np.array([data_points[labels == i].mean(axis=0)
                                    for i in range(self.number_clusters)])

            # Check for convergence
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tolerance):
                self.labels_ = labels
                break

            # Update centers for next iteration
            self.cluster_centers_ = new_centers
            
        # Set final iteration count
        self.iterations_ = iteration + 1

    def average_distance_to_centers(self, data_points):
        """
        Calculates the average distance to cluster centers.
        """

        # Check if model has been trained
        if self.cluster_centers_ is None or self.labels_ is None:
            raise ValueError("The model needs to be trained first using 'fit'.")

        total_distance = 0

        # Calculate distances with chosen metric
        if self.distance is None:

            for point, label in zip(data_points, self.labels_):
                
                # Get assigned cluster center
                center = self.cluster_centers_[label] 
                
                # Calculate euclidean distance
                dist = np.sqrt(np.sum((point - center) ** 2))
                
                # Add to total
                total_distance += dist

        else:

            for point, label in zip(data_points, self.labels_):

                center = self.cluster_centers_[label]

                # Use custom distance metric
                dist = self.distance(point, center)

                total_distance += dist

        # Return average distance
        return total_distance / len(data_points)
