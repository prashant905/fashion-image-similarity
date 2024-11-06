import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from datasketch import MinHashLSH
from model import DeepImageEncoder
from utils import create_minhash

class ImageSimilaritySearch:
    def __init__(self, image_folder, hash_dim=128, num_permutations=128):
        self.image_folder = image_folder
        self.hash_dim = hash_dim
        self.num_permutations = num_permutations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DeepImageEncoder(hash_dim=hash_dim).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.vis_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.lsh_index = MinHashLSH(threshold=0.8, num_perm=num_permutations)
        self.image_paths = []
        self.deep_features = []

    def _extract_deep_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def index_images(self):
        print("Indexing images...")
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        for image_name in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(self.image_folder, image_name)
            features = self._extract_deep_features(image_path)
            if features is not None:
                minhash = create_minhash(features, self.num_permutations)
                self.image_paths.append(image_path)
                self.deep_features.append(features)
                self.lsh_index.insert(image_path, minhash)
        
        self.deep_features = np.array(self.deep_features)
        print(f"Successfully indexed {len(self.image_paths)} images")

    def visualize_results(self, query_image_path, similar_images, figsize=(15, 10)):
        plt.figure(figsize=figsize)
        n = len(similar_images) + 1
        plt.subplot(2, (n+1)//2, 1)
        query_img = Image.open(query_image_path).convert('RGB')
        plt.imshow(query_img)
        plt.title('Query Image', fontsize=12)
        plt.axis('off')
        for idx, (path, similarity) in enumerate(similar_images, 1):
            plt.subplot(2, (n+1)//2, idx + 1)
            img = Image.open(path).convert('RGB')
            plt.imshow(img)
            plt.title(f'Similarity: {similarity:.3f}', fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def find_similar_images(self, query_image_path, num_results=5, visualize=True):
        query_features = self._extract_deep_features(query_image_path)
        query_minhash = create_minhash(query_features, self.num_permutations)
        lsh_candidates = self.lsh_index.query(query_minhash)
        if len(lsh_candidates) < num_results:
            similarities = np.dot(normalize(self.deep_features), normalize(query_features.reshape(1, -1)).T).flatten()
            top_indices = np.argsort(similarities)[::-1][:num_results]
            results = [(self.image_paths[idx], similarities[idx]) for idx in top_indices]
        else:
            candidate_features = []
            candidate_paths = []
            for image_path in lsh_candidates:
                idx = self.image_paths.index(image_path)
                candidate_features.append(self.deep_features[idx])
                candidate_paths.append(image_path)
            candidate_features = np.array(candidate_features)
            similarities = np.dot(normalize(candidate_features), normalize(query_features.reshape(1, -1)).T).flatten()
            sorted_indices = np.argsort(similarities)[::-1][:num_results]
            results = [(candidate_paths[idx], similarities[idx]) for idx in sorted_indices]
        if visualize:
            self.visualize_results(query_image_path, results)
        return results