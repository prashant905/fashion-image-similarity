# fashion_similarity/image_processing.py

import cv2
from PIL import Image
import numpy as np

class ImageProcessing:
    def __init__(self, resize_shape=(224, 224)):
        """
        Initialize the ImageProcessing class with a target resize shape.

        :param resize_shape: Tuple specifying the target size for resizing images.
        """
        self.resize_shape = resize_shape

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the input image to the target shape.

        :param image: Input image as a NumPy array.
        :return: Resized image as a NumPy array.
        """
        if image is None:
            print("Error: Received None as the image to resize.")
            return None
        resized_image = cv2.resize(image, self.resize_shape)
        return resized_image

    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image from BGR (OpenCV format) to RGB format.

        :param image: Input image in BGR format.
        :return: Image in RGB format.
        """
        if image is None:
            print("Error: Received None as the image to convert.")
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load, resize, and convert an image to RGB format.

        :param image_path: Path to the image file.
        :return: Preprocessed image as a NumPy array.
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            # Resize image
            image = image.resize(self.resize_shape)
            # Convert image to NumPy array
            image_array = np.array(image)
            return image_array
        except FileNotFoundError:
            print(f"File {image_path} not found.")
        except Exception as e:
            print(f"An error occurred while preprocessing image {image_path}: {e}")
        return None
