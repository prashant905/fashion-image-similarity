import os
from config import IMAGE_DIR

class DatasetLoader:
    def __init__(self, image_paths):
        self.image_paths = [os.path.join(IMAGE_DIR, img) for img in image_paths]

    def get_image_paths(self):
        return self.image_paths
