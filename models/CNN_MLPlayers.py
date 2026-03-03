import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

class CNNWithSizeMLPlayers(nn.Module):
    def __init__(self, num_classes=10, size_dim=2, pretrained=True):
        """
        To replicate Erik's model Deep Network Designer graph:
        - Use a pretrained ResNet backbone (e.g., ResNet-18) for image feature extraction.
        - Feature input size is 2 (majoraxis and minoraxis), which will be concatenated with the image features before classification.
        - Flatten 
        - Concatenate the image features with the size features (majoraxis and minoraxis).
        - MLP output changed from 1000 to num_classes (=10)
        """
        
        super().__init__()
        
        # Pretrained ResNet backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        self.backbone.fc = nn.Identity()  # Remove the final classification layer, replace with identity to get features ("do nothing" layer)
        # Backbone returns flattened 512-dim feature vector
        #self.fc = nn.Linear(512 + size_dim, num_classes)  # Final classifier input: image features + size features
        
        self.fc = nn.Sequential(
        nn.Linear(512 + size_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
)
        
    def forward(self, x_img, x_size):
        img_features = self.backbone(x_img)  # Extract image features
        combined_features = torch.cat((img_features, x_size), dim=1)  # Combine features
        logits = self.fc(combined_features)  # Classify
        # output = self.classifier(combined_features)  # Classify
        
        return logits