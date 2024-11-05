# fashion_similarity/utils.py

import matplotlib.pyplot as plt
import cv2
from typing import Dict, Tuple
import numpy as np
import os

def display_images(image_dict: Dict[str, np.ndarray], rows: int = 1, cols: int = 1, figsize: Tuple[int, int] = (12, 12)):
    """
    Display images in a grid format.

    :param image_dict: Dictionary where keys are titles and values are images as NumPy arrays.
    :param rows: Number of rows in the display grid.
    :param cols: Number of columns in the display grid.
    :param figsize: Tuple specifying figure size.
    """
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, (title, image) in enumerate(image_dict.items()):
        if idx < len(axes):
            axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(title)
            axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

def ensure_directory_exists(directory_path: str):
    """
    Ensure that a directory exists, and create it if it does not.

    :param directory_path: Path to the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created at: {directory_path}")
    else:
        print(f"Directory already exists at: {directory_path}")

def save_embeddings_to_file(embeddings: np.ndarray, file_path: str):
    """
    Save embeddings to a file in NumPy format.

    :param embeddings: 2D NumPy array of embeddings.
    :param file_path: File path to save the embeddings.
    """
    try:
        np.save(file_path, embeddings)
        print(f"Embeddings saved to {file_path}.")
    except Exception as e:
        print(f"Failed to save embeddings: {e}")

def load_embeddings_from_file(file_path: str) -> np.ndarray:
    """
    Load embeddings from a NumPy file.

    :param file_path: File path to load the embeddings from.
    :return: 2D NumPy array of embeddings.
    """
    try:
        embeddings = np.load(file_path)
        print(f"Embeddings loaded from {file_path}.")
        return embeddings
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return None
