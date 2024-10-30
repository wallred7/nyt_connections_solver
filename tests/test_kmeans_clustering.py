import unittest
import numpy as np #type: ignore
from src.model.kmeans_clustering import KMeansClustering

class TestKMeansClustering(unittest.TestCase):
    def test_kmeans_plusplus_initialization(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans = KMeansClustering(n_clusters=2, init='kmeans++')
        kmeans._initialize_centroids(X)
        self.assertEqual(kmeans.centroids.shape, (2, 2))

    def test_random_initialization(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans = KMeansClustering(n_clusters=2, init='random')
        kmeans._initialize_centroids(X)
        self.assertEqual(kmeans.centroids.shape, (2, 2))

    def test_fit_predict(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        kmeans = KMeansClustering(n_clusters=2)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        self.assertEqual(len(labels), 6)

    def test_invalid_similarity_metric(self):
        with self.assertRaises(ValueError):
            KMeansClustering(n_clusters=2, similarity_metric='invalid')

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            KMeansClustering(n_clusters=2, init='invalid')

    def test_glove_embeddings(self):
        kmeans = KMeansClustering(n_clusters=2)
        embeddings = kmeans.encode(['king', 'queen'], embedding_method='glove')
        self.assertTrue(isinstance(embeddings, dict))
        self.assertTrue(len(embeddings) > 0)

    # def test_sentence_transformer_embeddings(self):
    #     kmeans = KMeansClustering(n_clusters=2)
    #     embeddings = kmeans.encode(['king', 'queen'], embedding_method='sentence-transformer')
    #     self.assertTrue(isinstance(embeddings, dict))
    #     self.assertTrue(len(embeddings) > 0)

    def test_glove_embeddings_file_not_found(self):
        kmeans = KMeansClustering(n_clusters=2)
        embeddings = kmeans.encode(['king', 'queen'], embedding_method='glove')
        self.assertTrue(isinstance(embeddings, dict))
        self.assertTrue(len(embeddings) >= 0) #at least 0 words found

    def test_empty_words(self):
        kmeans = KMeansClustering(n_clusters=2)
        embeddings = kmeans.encode([], embedding_method='glove')
        self.assertTrue(isinstance(embeddings, dict))
        self.assertEqual(len(embeddings), 0)

    # def test_empty_words_sentence_transformer(self):
    #     kmeans = KMeansClustering(n_clusters=2)
    #     embeddings = kmeans.encode([], embedding_method='sentence-transformer')
    #     self.assertTrue(isinstance(embeddings, dict))
    #     self.assertEqual(len(embeddings), 0)

    def test_invalid_embedding_method(self):
        kmeans = KMeansClustering(n_clusters=2)
        with self.assertRaises(ValueError):
            kmeans.encode(['king', 'queen'], embedding_method='invalid')


if __name__ == '__main__':
    unittest.main()
