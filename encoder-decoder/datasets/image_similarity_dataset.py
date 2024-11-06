from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ImageSimilarityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob('**/*.jpg')) + \
                           list(Path(image_dir).glob('**/*.jpeg')) + \
                           list(Path(image_dir).glob('**/*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, str(image_path)
