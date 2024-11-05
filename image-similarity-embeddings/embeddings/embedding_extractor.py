# fashion_similarity/embedding_extractor.py

import numpy as np
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

class EmbeddingExtractor:
    def __init__(self, model_type="resnet50"):
        """
        Initialize the EmbeddingExtractor with a specified model.

        :param model_type: The type of model to use for embeddings ("resnet50" or "efficientnet").
        """
        self.model_type = model_type.lower()
        self.model = self._load_model()
        self.preprocess_input = self._get_preprocessing_function()

    def _load_model(self):
        """
        Load the specified pretrained model for embedding extraction.

        :return: Pretrained model without the top layers.
        """
        if self.model_type == "resnet50":
            model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        elif self.model_type == "efficientnet":
            model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
        else:
            raise ValueError("Unsupported model type. Choose 'resnet50' or 'efficientnet'.")
        print(f"{self.model_type.capitalize()} model loaded successfully.")
        return model

    def _get_preprocessing_function(self):
        """
        Get the appropriate preprocessing function based on the model type.

        :return: Preprocessing function for the chosen model.
        """
        if self.model_type == "resnet50":
            return preprocess_resnet
        elif self.model_type == "efficientnet":
            return preprocess_efficientnet

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding for a given image.

        :param image: Preprocessed image as a NumPy array.
        :return: Embedding vector as a 1D NumPy array.
        """
        # Convert the image to a batch format
        img_array = np.expand_dims(img_to_array(image), axis=0)
        # Preprocess the image
        img_array = self.preprocess_input(img_array)
        # Predict embedding
        embedding = self.model.predict(img_array).flatten()
        return embedding

    def save_embeddings(self, embeddings, file_path):
        """
        Save the embeddings to a file.

        :param embeddings: Embeddings as a NumPy array or DataFrame.
        :param file_path: File path to save the embeddings.
        """
        try:
            np.save(file_path, embeddings)
            print(f"Embeddings saved to {file_path}.")
        except Exception as e:
            print(f"Failed to save embeddings: {e}")
