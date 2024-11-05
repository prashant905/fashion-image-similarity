# fashion_similarity/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize the Similarity class with precomputed embeddings.

        :param embeddings: 2D NumPy array of embeddings for all images.
        """
        self.embeddings = embeddings
        self.similarity_matrix = None

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute the cosine similarity matrix for the embeddings.

        :return: 2D NumPy array representing the cosine similarity matrix.
        """
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print("Similarity matrix computed successfully.")
        return self.similarity_matrix

    def get_similar_items(self, item_index: int, top_n: int = 5) -> list:
        """
        Retrieve the indices and similarity scores of the most similar items.

        :param item_index: Index of the item in the embeddings matrix.
        :param top_n: Number of most similar items to retrieve.
        :return: List of tuples containing item indices and similarity scores.
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Get the similarity scores for the specified item
        similarity_scores = list(enumerate(self.similarity_matrix[item_index]))
        # Sort by similarity score in descending order and exclude the item itself
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        return sorted_scores
