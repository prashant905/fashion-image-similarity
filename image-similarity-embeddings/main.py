# main.py

from fashion_similarity.config import METADATA_PATH, IMAGE_FOLDER, EMBEDDINGS_PATH, MODEL_TYPE, RESIZE_SHAPE, EVALUATION_TOP_N
from fashion_similarity.data_loader import DataLoader
from fashion_similarity.image_processing import ImageProcessing
from fashion_similarity.embedding_extractor import EmbeddingExtractor
from fashion_similarity.similarity import Similarity
from fashion_similarity.recommendation import RecommendationSystem
from fashion_similarity.evaluation import Evaluation
from fashion_similarity.utils import ensure_directory_exists, save_embeddings_to_file

import numpy as np

def main():
    # Step 1: Load Data
    data_loader = DataLoader(METADATA_PATH, IMAGE_FOLDER)
    fashion_data = data_loader.load_metadata()
    print(f"Data loaded: {fashion_data.shape[0]} entries")
    
    # Step 2: Process Images and Extract Embeddings
    image_processor = ImageProcessing(RESIZE_SHAPE)
    embedding_extractor = EmbeddingExtractor(model_type=MODEL_TYPE)
    
    embeddings = []
    missing_images = []
    
    for idx, row in fashion_data.iterrows():
        image_path = row['image_path']
        image = data_loader.load_image(image_path)
        
        if image is not None:
            resized_image = image_processor.resize_image(image)
            embedding = embedding_extractor.extract_embedding(resized_image)
            embeddings.append(embedding)
        else:
            missing_images.append(idx)
            embeddings.append([np.nan] * 2048)  # Placeholder for missing images
    
    embeddings = np.array(embeddings)
    embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]  # Remove rows with NaN values
    fashion_data = fashion_data.drop(missing_images).reset_index(drop=True)
    
    # Save embeddings
    ensure_directory_exists('data/embeddings')
    save_embeddings_to_file(embeddings, EMBEDDINGS_PATH)
    
    # Step 3: Compute Similarity Matrix
    similarity = Similarity(embeddings)
    similarity_matrix = similarity.compute_similarity_matrix()
    
    # Step 4: Initialize Recommendation System
    recommendation_system = RecommendationSystem(similarity, data_loader, image_processor)
    
    # Example: Get and visualize recommendations for a sample image
    sample_index = 0  # Change index to test with different images
    recommendation_system.visualize_recommendations(sample_index, top_n=DEFAULT_TOP_N)
    
    # Step 5: Evaluate Recommendations
    evaluation = Evaluation(recommendation_system, fashion_data)
    evaluation.generate_report(top_n=EVALUATION_TOP_N)  # Adjust top_n as needed
    
    print("Report generated successfully.")

if __name__ == "__main__":
    main()
