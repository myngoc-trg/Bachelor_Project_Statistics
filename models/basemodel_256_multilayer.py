import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models


class CNN256WithSizeMLP_multi(nn.Module):
    def __init__(self, num_classes=10, size_dim=2, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()   # output: [B, 512]

        self.embed_dim = 512
        self.proj_dim = 256

        # image feature projection: 512 -> 256
        self.image_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_dim),
            nn.ReLU()
        )

      
        self.classifier = nn.Sequential(
        nn.Linear(self.proj_dim + size_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_classes)
        )

        self.fc = self.classifier

    def embed(self, x_img):
        """
        Original backbone embedding: [B, 512]
        """
        return self.backbone(x_img)

    def embed_projected(self, x_img):
        """
        Projected image embedding: [B, 256]
        """
        img_features = self.backbone(x_img)
        return self.image_projection(img_features)

    def forward(self, x_img, x_size):
        img_features = self.backbone(x_img)                 # [B, 512]
        img_features_256 = self.image_projection(img_features)  # [B, 256]

        combined_features = torch.cat((img_features_256, x_size), dim=1)  # [B, 258]
        logits = self.classifier(combined_features)         # [B, 10]

        return logits