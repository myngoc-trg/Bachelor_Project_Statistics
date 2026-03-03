import torch
import torch.nn as nn

class BinarySpecialist(nn.Module):
    """
    Binary specialist for a pair. Uses stage1_model.embed(img) -> [B, D]
    """
    def __init__(self, stage1_model, size_dim=2, hidden=128, dropout=0.2, freeze_stage1=True):
        super().__init__()
        self.stage1 = stage1_model

        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad = False

        if not hasattr(self.stage1, "embed") or not hasattr(self.stage1, "embed_dim"):
            raise AttributeError("stage1_model must implement .embed(x_img) and have .embed_dim")

        D = int(self.stage1.embed_dim)

        """ 
        self.head = nn.Sequential(
            nn.Linear(D + size_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
        """
        self.head = nn.Linear(D + size_dim, 2)

    def forward(self, x_img, x_size):
        f = self.stage1.embed(x_img)           # [B, D]
        x = torch.cat([f, x_size], dim=1)      # [B, D+2]
        return self.head(x)                    # [B, 2]