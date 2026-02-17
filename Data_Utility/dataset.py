import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict

class PollenFolderWithSizeDataset(Dataset):
    def __init__(self, img_dir: str, class_to_idx: Dict[str, int], size_lookup: Dict[str, Tuple[float, float]], transform=None):
        """
        Initializes the dataset with the directory of images and a size lookup dictionary.

        Args:
            img_dir (str): The directory containing the images.
            class_to_idx (Dict[str, int]): A dictionary mapping class names to integer indices.
            size_lookup (Dict[str, Tuple[float, float]]): A dictionary mapping image names to their sizes (mijoraxis, minoraxis).
            transform: Optional transformations to be applied on the images.
            
        Returns:
            image tensor [3,W,H]
            size tensor [2]
            label tensor [1] or scalar
        """
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.size_lookup = size_lookup
        self.transform = transform
        
        # Build list of (image_path, label) tuples
        self.samples = self._scan_root()

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.samples)

    def _scan_root(self) -> List[Tuple[str, str]]:
        """Scans the root directory to find image files and their corresponding labels."""
        samples = []
        for class_name in sorted(os.listdir(self.img_dir)):
            class_dir = os.path.join(self.img_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for fn in sorted(os.listdir(class_dir)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    samples.append((os.path.join(class_dir, fn), class_name))
    
        return samples
    
    def __getitem__(self, idx: int):
        img_path, class_name = self.samples[idx]
        filename = os.path.basename(img_path)
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        if filename not in self.size_lookup:
            raise KeyError(f"Image filename '{filename}' not found in size lookup dictionary.")
        
        major, minor = self.size_lookup[filename]
        size = torch.tensor([major, minor], dtype=torch.float32)
        
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        
        return img, size, label