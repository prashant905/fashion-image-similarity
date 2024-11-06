import faiss
import numpy as np
from config import INDEX_PATH, IMAGE_PATHS_FILE

class FAISSIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        self.index.add(np.vstack(embeddings))

    def save_index(self):
        faiss.write_index(self.index, INDEX_PATH)

    def load_index(self):
        self.index = faiss.read_index(INDEX_PATH)

    def save_image_paths(self, image_paths):
        np.save(IMAGE_PATHS_FILE, np.array(image_paths))

    def load_image_paths(self):
        return np.load(IMAGE_PATHS_FILE, allow_pickle=True)

    def search(self, query_embedding, top_n):
        distances, indices = self.index.search(query_embedding, top_n)
        return distances, indices
