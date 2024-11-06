import torch
import torch.nn as nn
import torchvision.models as models

class DeepImageEncoder(nn.Module):
    def __init__(self, hash_dim=128):
        super(DeepImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.hash_layer = nn.Sequential(
            nn.Linear(2048, hash_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        return x