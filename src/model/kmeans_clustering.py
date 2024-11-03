import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances  # type: ignore
# from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class KMeansClustering:
    """
    K-Means clustering algorithm with multiple similarity metrics and word embedding encoding.
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, similarity_metric: str = 'cosine', init: str = 'random', embedding_dim: int = 50):
        """
        Initializes KMeansClustering with specified parameters.

        Args:
            n_clusters (int): The number of clusters.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            similarity_metric (str, optional): The similarity metric to use ('cosine' or 'euclidean'). Defaults to 'cosine'.
            init (str, optional): The centroid initialization method ('random' or 'kmeans++'). Defaults to 'random'.
            embedding_dim (int, optional): The pretrained embedding dimensions. Defaults to 50.

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
        self.embedding_dim = embedding_dim

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
            self.centroids = self._initialize_kmeans_plusplus_centroids(X)

    def _initialize_kmeans_plusplus_centroids(self, X: np.ndarray) -> np.ndarray:
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
            return -euclidean_distances(X, centroids)  # negate for max
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
        X = np.nan_to_num(X, nan=0.0) #Handle NaN values
    
        self._initialize_centroids(X)

        # K-means steps
        # 1. initialize centroids at a a given point in the same dimensional space 
        # 2. 

        for _ in range(self.max_iter):
            similarities = self._compute_similarity(X, self.centroids)
            
            labels = np.argmax(similarities, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids, rtol=1e-1024, atol=1e-1024):
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

    def _load_glove_embeddings(self, words: List[str], glove_path: str = 'src/model/glove/glove.6B.{dim}d.txt') -> np.ndarray:
        """
        Load GloVe embeddings for a given list of words. Only reads in the words from the list.

        Args:
            words: List of words to get embeddings for
            glove_path: Path to GloVe file (e.g., 'glove.6B.100d.txt')
            embedding_dim: Dimension of embeddings (default: 100)

        Returns:
            NumPy array of word embeddings
        """
        if not words.all(): # Handle empty words list
            return np.empty((0, self.embedding_dim))
        words_set = set(map(str.lower, words))  # O(1) lookup
        word_to_embedding = {}

        glove_path_dim = glove_path.format(dim=self.embedding_dim)

        try:
            with open(glove_path_dim, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0].lower()

                    if word in words_set:
                        vector = np.asarray(values[1:], dtype='float32')
                        word_to_embedding[word] = vector
        except FileNotFoundError:
            print(f"Error: GloVe file not found at {glove_path_dim}")
            return np.empty((0, self.embedding_dim))
        except Exception as e:
            print(f"An unexpected error occurred while reading the GloVe file: {e}")
            return np.empty((0, self.embedding_dim))

        # Report coverage
        found_words = set(word_to_embedding.keys())
        missing_words = words_set - found_words
        coverage = len(found_words) / len(words_set) * 100 if words_set else 0 #Handle empty words_set

        print(f"Found embeddings for {len(found_words)}/{len(words_set)} words ({coverage:.1f}%)")
        if missing_words:
            print(f"Missing words: {', '.join(list(missing_words)[:10])}",
                  "..." if len(missing_words) > 10 else "")

        embeddings = np.array(list(word_to_embedding.values()))
        return embeddings

    def _load_sentence_transformer_embeddings(self, words: List[str], model_name: str = 'all-mpnet-base-v2') -> np.ndarray:
        """
        Encodes words using word embeddings from a specified SentenceTransformer model.

        Args:
            words (List[str]): The words to encode.
            model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to 'all-mpnet-base-v2'.

        Returns:
            NumPy array of word embeddings

        Raises:
            Exception: If there is an error during model loading or encoding.
        """
        try:
            model = SentenceTransformer(model_name) 
            embeddings = model.encode(words)
            return embeddings
        except Exception as e:
            raise Exception(f"Error during word embedding encoding: {e}")

    def encode(self, words: List[str], embedding_method: str = 'glove') -> np.ndarray:
        """
        Encodes words using either GloVe or SentenceTransformer embeddings.

        Args:
            words (List[str]): The words to encode.
            embedding_method (str, optional): The embedding method to use ('glove' or 'sentence-transformer'). Defaults to 'glove'.

        Returns:
            NumPy array of word embeddings

        Raises:
            ValueError: If an invalid embedding method is specified.
        """
        if embedding_method == 'glove':
            return self._load_glove_embeddings(words)
        elif embedding_method == 'sentence-transformer':
            return self._load_sentence_transformer_embeddings(words)
        else:
            raise ValueError("Invalid embedding method. Choose 'glove' or 'sentence-transformer'.")
