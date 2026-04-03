import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtTiny384WithSize(nn.Module):
    def __init__(
        self,
        num_classes=10,
        size_dim=2,
        pretrained=True,
        head_hidden=256,
        dropout=0.2
    ):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # Feature extractor: outputs [B, C, H, W]
        self.features = convnext.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Original ConvNeXt embedding dimension
        self.backbone_dim = convnext.classifier[-1].in_features   # usually 768

        # Final reduced embedding dimension = half of original
        self.embed_dim = self.backbone_dim // 2                   # 384

        # Dense projection: 768 -> 384
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, self.embed_dim),
            nn.ReLU()
        )

        # Classifier after fusion with size vector
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim + size_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes)
        )

        self.fc = self.classifier

    def embed_backbone(self, x_img):
        """
        Original ConvNeXt embedding before projection.
        Shape: [B, 768]
        """
        x = self.features(x_img)          # [B, C, H, W]
        x = self.avgpool(x)               # [B, C, 1, 1]
        x = torch.flatten(x, 1)           # [B, 768]
        return x

    def embed(self, x_img):
        """
        Reduced dense embedding after projection.
        Shape: [B, 384]
        """
        x = self.embed_backbone(x_img)    # [B, 768]
        x = self.projection(x)            # [B, 384]
        return x

    def forward(self, x_img, x_size):
        img_feat = self.embed(x_img)      # [B, 384]
        fused = torch.cat([img_feat, x_size], dim=1)   # [B, 384 + size_dim]
        logits = self.classifier(fused)
        return logits