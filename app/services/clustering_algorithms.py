"""
Implementierung von K-Means Clustering mit benutzerdefinierten Distanzmetriken.
"""

import numpy as np
from sklearn.cluster import KMeans


# Distanz-Funktionen
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))


def minkowski_distance(x, y, p=2):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Custom KMeans mit Unterstützung für verschiedene Distanzmaße


class CustomKMeans:
    def __init__(self, n_clusters, distance_metric="EUCLIDEAN", max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.iterations_ = 0

        if distance_metric == "EUCLIDEAN":
            self.distance = euclidean_distance
        elif distance_metric == "MANHATTAN":
            self.distance = manhattan_distance
        elif distance_metric == "CHEBYSHEV":
            self.distance = chebyshev_distance
        elif distance_metric == "MINKOWSKI":
            self.distance = minkowski_distance
        else:
            raise ValueError(f"Unbekanntes Distanzmaß: {distance_metric}")

    def fit(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters,
                        init='k-means++', max_iter=1, n_init=1)
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_

        for iteration in range(self.max_iter):
            dists = np.array([[self.distance(x, center)
                             for center in self.cluster_centers_] for x in X])
            labels = np.argmin(dists, axis=1)

            new_centers = np.array([X[labels == i].mean(axis=0)
                                   for i in range(self.n_clusters)])

            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                self.labels_ = labels
                break

            self.cluster_centers_ = new_centers
            self.labels_ = labels

        self.iterations_ = iteration + 1
