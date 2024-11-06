import os

# Base directory for images and data
BASE_DIR = 'fashion-dataset'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

# Paths for FAISS index and image paths file
INDEX_PATH = os.path.join(BASE_DIR, 'image_embeddings.index')
IMAGE_PATHS_FILE = os.path.join(BASE_DIR, 'image_paths.npy')

# Example query image
QUERY_IMAGE_PATH = os.path.join(IMAGE_DIR, '15970.jpg')
