import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

class CNNWithSizeMLP(nn.Module):
    def __init__(self, num_classes=10, size_dim=2, pretrained=True):
        super().__init__()
        
        # Pretrained ResNet backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        self.fc = nn.Linear(512 + 2, num_classes)  # Final classifier input: image features + size features
        
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 + size_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
        '''
        # Final classifier
        
    def forward(self, x_img, x_size):
        img_features = self.backbone(x_img)  # Extract image features
        combined_features = torch.cat((img_features, x_size), dim=1)  # Combine features
        logits = self.fc(combined_features)  # Classify
        # output = self.classifier(combined_features)  # Classify
        
        return logits