from config import QUERY_IMAGE_PATH, INDEX_PATH, IMAGE_PATHS_FILE
from src.dataset import DatasetLoader
from src.embedding_extractor import EmbeddingExtractor
from src.faiss_index import FAISSIndex
from src.display import display_top_similar_images
import numpy as np

def main():
    # Load dataset and extractor
    dataset_loader = DatasetLoader(image_paths=data['image_path'])
    embedding_extractor = EmbeddingExtractor()
    
    embeddings = []
    image_names = dataset_loader.get_image_paths()
    
    for image_path in image_names:
        embeddings.append(embedding_extractor.get_image_embedding(image_path))
    
    # Initialize and populate FAISS index
    index = FAISSIndex(dimension=embeddings[0].shape[0])
    index.add_embeddings(embeddings)
    index.save_index()
    index.save_image_paths(image_names)

    # Query and display
    query_embedding = embedding_extractor.get_image_embedding(QUERY_IMAGE_PATH).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n=5)
    
    image_paths = index.load_image_paths()
    similar_images = [(image_paths[idx], 1 - (dist / 2)) for idx, dist in zip(indices[0], distances[0])]
    display_top_similar_images(QUERY_IMAGE_PATH, similar_images)

if __name__ == "__main__":
    main()
