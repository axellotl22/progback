"""
Implementierung von K-Means Clustering mit benutzerdefinierten Distanzmetriken.
"""

import numpy as np
from sklearn.cluster import KMeans

# Distanz-Funktionen
def euclidean_distance(point_a, point_b):
    """Berechne den euklidischen Abstand zwischen zwei Punkten."""
    return np.sqrt(np.sum((point_a - point_b) ** 2))

def squared_euclidean_distance(point_a, point_b):
    """Berechne den quadratischen euklidischen Abstand zwischen zwei Punkten."""
    return np.sum((point_a - point_b) ** 2)

def manhattan_distance(point_a, point_b):
    """Berechne den Manhattan Abstand zwischen zwei Punkten."""
    return np.sum(np.abs(point_a - point_b))

def chebyshev_distance(point_a, point_b):
    """Berechne den Chebyshev Abstand zwischen zwei Punkten."""
    return np.max(np.abs(point_a - point_b))

def canberra_distance(point_a, point_b):
    """Berechne den Canberra Abstand zwischen zwei Punkten."""
    return np.sum(np.abs(point_a - point_b) / (np.abs(point_a) + np.abs(point_b)))

def chi_square_distance(point_a, point_b):
    """Berechne den Chi-Quadrat Abstand zwischen zwei Punkten."""
    return 0.5 * np.sum((point_a - point_b) ** 2 / (point_a + point_b + 1e-10))

def jaccards_distance(point_a, point_b):
    """Berechne den Jaccard Abstand zwischen zwei Punkten."""
    intersection = np.minimum(point_a, point_b).sum()
    union = np.maximum(point_a, point_b).sum()
    return 1 - (intersection / union)

class CustomKMeans:
    """Implementiere KMeans Clustering mit verschiedenen Distanzmaßen."""

    # Klassenvariable für unterstützte Distanzmetriken
    SUPPORTED_DISTANCE_METRICS = {
        "EUCLIDEAN": euclidean_distance,
        "SQUARED_EUCLIDEAN": squared_euclidean_distance,
        "MANHATTAN": manhattan_distance,
        "CHEBYSHEV": chebyshev_distance,
        "CANBERRA": canberra_distance,
        "CHI_SQUARE": chi_square_distance,
        "JACCARDS": jaccards_distance
    }

    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.iterations_ = 0

        # Wähle die Distanz-Funktion basierend auf dem gegebenen Distanzmaß von der Klassenvariable
        self.distance = self.SUPPORTED_DISTANCE_METRICS.get(distance_metric)
        if self.distance is None:
            raise ValueError(f"Unbekanntes Distanzmaß: {distance_metric}")

    def fit(self, data_points):
        """Trainiere das Modell mit den gegebenen Daten."""
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(data_points)
        self.cluster_centers_ = kmeans.cluster_centers_

        for iteration in range(self.max_iter):
            dists = np.array([[self.distance(point, center)
                             for center in self.cluster_centers_] for point in data_points])
            labels = np.argmin(dists, axis=1)

            new_centers = np.array([data_points[labels == i].mean(axis=0)
                                   for i in range(self.n_clusters)])

            # Überprüfung, ob die Konvergenz erreicht ist
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                self.labels_ = labels
                break

            self.cluster_centers_ = new_centers
            self.labels_ = labels

        self.iterations_ = iteration + 1
        
    def average_distance_to_centers(self, data_points):
        """Berechne den durchschnittlichen Abstand aller Datenpunkte zu ihren Clusterzentren."""
        if self.cluster_centers_ is None or self.labels_ is None:
            raise ValueError("Das Modell muss zuerst mit 'fit' trainiert werden.")

        total_distance = 0

        # Wenn der euklidische Abstand als Standard gesetzt ist
        if self.distance is None:
            for point, label in zip(data_points, self.labels_):
                center = self.cluster_centers_[label]
                total_distance += np.sqrt(np.sum((point - center) ** 2))
        else:
            for point, label in zip(data_points, self.labels_):
                center = self.cluster_centers_[label]
                total_distance += self.distance(point, center)

        return total_distance / len(data_points)
