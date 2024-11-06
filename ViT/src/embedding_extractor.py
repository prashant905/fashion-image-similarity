from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch

class EmbeddingExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

    def get_image_embedding(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0].detach().numpy().astype('float32')
