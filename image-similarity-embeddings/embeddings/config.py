# fashion_similarity/config.py

# Paths
METADATA_PATH = 'data/fashion-dataset/styles.csv'
IMAGE_FOLDER = 'data/fashion-dataset/images'
EMBEDDINGS_PATH = 'data/embeddings/fashion_embeddings.npy'

# Image Processing
RESIZE_SHAPE = (224, 224)

# Model Settings
MODEL_TYPE = "resnet50"

# Recommendation System
DEFAULT_TOP_N = 5

# Evaluation Settings
EVALUATION_TOP_N = 1
