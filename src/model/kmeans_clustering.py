import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances #type: ignore 
from sentence_transformers import SentenceTransformer, util
from typing import List, Union, Tuple, Optional

class KMeansClustering:
    """
    K-Means clustering algorithm with multiple similarity metrics and word embedding encoding.
    """
    def __init__(self, n_clusters: int, max_iter: int = 100, similarity_metric: str = 'cosine', init: str = 'random'):
        """
        Initializes KMeansClustering with specified parameters.

        Args:
            n_clusters (int): The number of clusters.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            similarity_metric (str, optional): The similarity metric to use ('cosine' or 'euclidean'). Defaults to 'cosine'.
            init (str, optional): The centroid initialization method ('random' or 'kmeans++'). Defaults to 'random'.

        Raises:
            ValueError: If an invalid similarity metric or initialization method is provided.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if similarity_metric not in ['cosine', 'euclidean']:
            raise ValueError("Invalid similarity metric. Choose 'cosine' or 'euclidean'.")
        if init not in ['random', 'kmeans++']:
            raise ValueError("Invalid initialization method. Choose 'random' or 'kmeans++'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.similarity_metric = similarity_metric
        self.init = init
        self.centroids = None

    def _initialize_centroids(self, X: np.ndarray) -> None:
        """
        Initializes centroids using either random sampling or k-means++.

        Args:
            X (np.ndarray): The data points.
        """
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]
        elif self.init == 'kmeans++':
            self.centroids = self._kmeans_plusplus_init(X)

    def _initialize_probabalistic_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initializes centroids using the k-means++ algorithm.

        Args:
            X (np.ndarray): The data points.

        Returns:
            np.ndarray: The initialized centroids.
        """
        centroids = []
        centroids.append(X[np.random.randint(0, X.shape[0])])
        for _ in range(self.n_clusters - 1):
            distances = np.min(euclidean_distances(X, centroids), axis=1)
            probabilities = distances / np.sum(distances)
            next_centroid_index = np.random.choice(X.shape[0], p=probabilities)
            centroids.append(X[next_centroid_index])
        return np.array(centroids)


    def _compute_similarity(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Computes the similarity between data points and centroids.

        Args:
            X (np.ndarray): The data points.
            centroids (np.ndarray): The centroids.

        Returns:
            np.ndarray: The similarity matrix.

        Raises:
            ValueError: If an invalid similarity metric is used.
        """
        if self.similarity_metric == 'cosine':
            return cosine_similarity(X, centroids)
        elif self.similarity_metric == 'euclidean':
            return -euclidean_distances(X, centroids) #negate for max
        else:
            raise ValueError("Invalid similarity metric.")

    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Fits the KMeansClustering model to the data.

        Args:
            X (np.ndarray): The data points.

        Returns:
            KMeansClustering: The fitted model.
        """
        self._initialize_centroids(X)
        for _ in range(self.max_iter):
            similarities = self._compute_similarity(X, self.centroids)
            labels = np.argmax(similarities, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for the given data points.

        Args:
            X (np.ndarray): The data points.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        similarities = self._compute_similarity(X, self.centroids)
        return np.argmax(similarities, axis=1)


    def encode_words(self, words: List[str], model_name: str = 'all-mpnet-base-v2') -> np.ndarray:
        """
        Encodes words using word embeddings from a specified SentenceTransformer model.

        Args:
            words (List[str]): The words to encode.
            model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to 'all-mpnet-base-v2'.

        Returns:
            np.ndarray: The encoded words.

        Raises:
            Exception: If there is an error during model loading or encoding.
        """
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(words)
            return embeddings
        except Exception as e:
            raise Exception(f"Error during word embedding encoding: {e}")
