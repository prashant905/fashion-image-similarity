# fashion_similarity/data_loader.py

import pandas as pd
import cv2
from pathlib import Path

class DataLoader:
    def __init__(self, metadata_path: str, image_folder: str):
        """
        Initialize the DataLoader with paths to metadata and images.

        :param metadata_path: Path to the CSV file containing image metadata.
        :param image_folder: Path to the folder containing images.
        """
        self.metadata_path = metadata_path
        self.image_folder = Path(image_folder)
        self.data = None

    def load_metadata(self):
        """
        Load metadata from a CSV file and add an image path column.
        
        :return: Pandas DataFrame with metadata.
        """
        try:
            # Load CSV file and handle any bad lines
            self.data = pd.read_csv(self.metadata_path, on_bad_lines='skip')
            self.data['image_path'] = self.data.apply(lambda row: str(row['id']) + '.jpg', axis=1)
            self.data = self.data.reset_index(drop=True)
            print("Metadata loaded successfully.")
        except FileNotFoundError:
            print(f"File {self.metadata_path} not found.")
        except Exception as e:
            print(f"An error occurred while loading metadata: {e}")

        return self.data

    def get_image_path(self, image_name: str) -> str:
        """
        Construct the full path to an image.

        :param image_name: Name of the image file.
        :return: Full path to the image.
        """
        return str(self.image_folder / image_name)

    def load_image(self, image_name: str):
        """
        Load an image from the image folder.

        :param image_name: Name of the image file.
        :return: Loaded image as a NumPy array.
        """
        image_path = self.get_image_path(image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_name} not found.")
        return image
