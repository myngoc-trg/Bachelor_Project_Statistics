import torch
from torch.utils.data import Dataset

class BinaryPairDataset(Dataset):
    """
    Wraps an existing dataset that yields (img, size, label).
    Keeps only labels in {class_a, class_b} and remaps to {0,1}.
    """
    def __init__(self, base_dataset, class_a, class_b):
        self.base = base_dataset
        self.class_a = int(class_a)
        self.class_b = int(class_b)
        self.map = {self.class_a: 0, self.class_b: 1}

        self.indices = []
        for i in range(len(base_dataset)):
            _, _, y = base_dataset[i]
            y = int(y) if not torch.is_tensor(y) else int(y.item())
            if y in self.map:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img, size, y = self.base[i]
        y = int(y) if not torch.is_tensor(y) else int(y.item())
        y_bin = self.map[y]
        return img, size, torch.tensor(y_bin, dtype=torch.long)