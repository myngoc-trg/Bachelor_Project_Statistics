import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtTinyLinearWithSize(nn.Module):
    def __init__(self, num_classes=10, size_dim=2, pretrained=True, head_hidden=256, dropout=0.2):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # Keep only the feature extractor part
        self.features = convnext.features  # outputs [B, C, H, W]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ConvNeXt-Tiny last channel dim (should be 768)
        self.embed_dim = convnext.classifier[-1].in_features
        
        self.classifier = nn.Linear(self.embed_dim + size_dim, num_classes)



        self.fc = self.classifier

    def embed(self, x_img):
        x = self.features(x_img)          # [B, C, H, W]
        x = self.avgpool(x)               # [B, C, 1, 1]
        return torch.flatten(x, 1)        # [B, C]
            
    def forward(self, x_img, x_size):
        x = self.features(x_img)          # [B, C, H, W]
        x = self.avgpool(x)               # [B, C, 1, 1]
        img_feat = torch.flatten(x, 1)    # [B, C]

        fused = torch.cat([img_feat, x_size], dim=1)  # [B, C+2]
        logits = self.classifier(fused)
        return logits