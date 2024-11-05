# fashion_similarity/recommendation.py

import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from .similarity import Similarity
from .data_loader import DataLoader
from .image_processing import ImageProcessing

class RecommendationSystem:
    def __init__(self, similarity_matrix: Similarity, data_loader: DataLoader, image_processor: ImageProcessing):
        """
        Initialize the RecommendationSystem with similarity matrix, data loader, and image processor.

        :param similarity_matrix: Instance of the Similarity class with precomputed similarity matrix.
        :param data_loader: Instance of DataLoader to load images.
        :param image_processor: Instance of ImageProcessing to process images for display.
        """
        self.similarity = similarity_matrix
        self.data_loader = data_loader
        self.image_processor = image_processor

    def get_recommendations(self, image_index: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Get recommended image indices and their similarity scores.

        :param image_index: Index of the image for which recommendations are needed.
        :param top_n: Number of recommendations to return.
        :return: List of tuples containing recommended image indices and similarity scores.
        """
        return self.similarity.get_similar_items(image_index, top_n=top_n)

    def visualize_recommendations(self, image_index: int, top_n: int = 5):
        """
        Display the input image along with its top recommended similar images.

        :param image_index: Index of the image for which recommendations are needed.
        :param top_n: Number of recommendations to visualize.
        """
        # Get the recommendations
        recommended_indices, recommended_scores = zip(*self.get_recommendations(image_index, top_n))

        # Load and display the input image
        input_image_path = self.data_loader.data.iloc[image_index].image_path
        input_image = self.data_loader.load_image(input_image_path)
        input_image_rgb = self.image_processor.convert_to_rgb(input_image)
        
        # Display the input image
        plt.figure(figsize=(12, 12))
        plt.subplot(2, top_n + 1, 1)
        plt.imshow(input_image_rgb)
        plt.title("Input Image")
        plt.axis("off")

        # Display recommended images
        for idx, rec_index in enumerate(recommended_indices):
            recommended_image_path = self.data_loader.data.iloc[rec_index].image_path
            recommended_image = self.data_loader.load_image(recommended_image_path)
            recommended_image_rgb = self.image_processor.convert_to_rgb(recommended_image)
            
            plt.subplot(2, top_n + 1, idx + 2)
            plt.imshow(recommended_image_rgb)
            plt.title(f"Score: {recommended_scores[idx]:.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
