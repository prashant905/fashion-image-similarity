import torch
import faiss
import numpy as np
from torchvision import transforms
from .autoencoder import ImageAutoencoder
from datasets.image_similarity_dataset import ImageSimilarityDataset
from torch.utils.data import DataLoader

class ImageSimilaritySearch:
    def __init__(self, latent_dim=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.latent_dim = latent_dim
        self.model = ImageAutoencoder(latent_dim).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.index = None
        self.image_paths = []

    def train(self, image_dir, epochs=10, batch_size=32, learning_rate=0.001):
        dataset = ImageSimilarityDataset(image_dir, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_images, _ in dataloader:
                batch_images = batch_images.to(self.device)

                # Forward pass
                reconstructed, _ = self.model(batch_images)
                loss = criterion(reconstructed, batch_images)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    def build_index(self, image_dir):
        dataset = ImageSimilarityDataset(image_dir, self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_embeddings = []
        self.image_paths = []

        with torch.no_grad():
            for batch_images, batch_paths in dataloader:
                batch_images = batch_images.to(self.device)
                _, latent = self.model(batch_images)
                all_embeddings.append(latent.cpu().numpy())
                self.image_paths.extend(batch_paths)

        all_embeddings = np.vstack(all_embeddings)

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.latent_dim)
        self.index.add(all_embeddings.astype('float32'))

    def search(self, query_image_path, k=5):
        query_image = Image.open(query_image_path).convert('RGB')
        query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)

        # Get query embedding
        self.model.eval()
        with torch.no_grad():
            _, query_embedding = self.model(query_tensor)
            query_embedding = query_embedding.cpu().numpy()

        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'image_path': self.image_paths[idx],
                'distance': float(dist)
            })

        return results

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'image_paths': self.image_paths
        }, path + '_model.pth')

        if self.index is not None:
            faiss.write_index(self.index, path + '_index.faiss')

    def load_model(self, path):
        checkpoint = torch.load(path + '_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.image_paths = checkpoint['image_paths']

        if Path(path + '_index.faiss').exists():
            self.index = faiss.read_index(path + '_index.faiss')
