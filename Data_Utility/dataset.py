from email.policy import default
import os
from sklearn.base import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict

class PollenFolderWithSizeDataset(Dataset):
    def __init__(self, img_dir: str, class_to_idx: Dict[str, int], size_lookup: Dict[str, Tuple[float, float]], species_mean_lookup: Dict[str, Tuple[float, float]], fill_missing_bool: bool = False, transform=None, print_summary: bool = True):
        """
        Initializes the dataset with the directory of images and a size lookup dictionary.

        Args:
            img_dir (str): The directory containing the images.
            class_to_idx (Dict[str, int]): A dictionary mapping class names to integer indices.
            size_lookup (Dict[str, Tuple[float, float]]): A dictionary mapping image names to their sizes (mijoraxis, minoraxis).
            species_mean_lookup (Dict[str, Tuple[float, float]]): A dictionary mapping species names to their mean sizes.
            global_mean (Tuple[float, float]): The overall mean size across all species.
            transform: Optional transformations to be applied on the images.
            
        Returns:
            image tensor [3,W,H]
            size tensor [2]
            label tensor [1] or scalar
        """
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.size_lookup = size_lookup
        self.species_mean_lookup = species_mean_lookup
        self.transform = transform
        self.fill_missing_bool = fill_missing_bool
        
        # Build list of (image_path, label) tuples
        all_samples = self._scan_root()
        
        # Count missing size entries for reporting
        self.missing_size_total_count = 0
        
        # Per-species missing count for detailed reporting
        self.missing_size_species = defaultdict(int)
        
        self._summary_printed = False  # Flag to ensure summary is printed only once
        
        self.samples = []
        
        for img_path, class_name in all_samples:
            filename = os.path.basename(img_path)
            if filename in self.size_lookup:
                self.samples.append((img_path, class_name))
                continue
            
            self.missing_size_total_count += 1
            self.missing_size_species[class_name] += 1
            
            if not self.fill_missing_bool:
                continue
            
            self.samples.append((img_path, class_name)) # keep the sample even if size is missing, will handle in __getitem__ with filling logic
            
        
        if print_summary:
            print(f"\nInitialized dataset from '{self.img_dir}' with {len(self.samples)} samples.")
            if self.missing_size_total_count > 0:
                self.print_missing_summary()
            else:
                print(f"\nNo missing size data found in the image dataset of {self.img_dir}.")
        
    
    def __len__(self):
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
        
        '''
        if filename not in self.size_lookup:
            raise KeyError(f"Image filename '{filename}' not found in size lookup dictionary.")
        
        major, minor = self.size_lookup[filename]
        ''' 

        if filename in self.size_lookup:
            major, minor = self.size_lookup[filename]
        else:
            # Fill in species mean from the same species if available
            #print(f"\nWarning: Size data missing for image '{filename}'. Attempting to fill with species mean.")
            
                if class_name in self.species_mean_lookup:
                    major, minor = self.species_mean_lookup[class_name]
                else:
                    print(f"\nWarning: Size data missing for image '{filename}' and no species mean available for '{class_name}'. Skipping sample.")
       
        size = torch.tensor([major, minor], dtype=torch.float32)
        
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)   
        
        return img, size, label
    
    def print_missing_summary(self, top_k: int | None = None):
        """Print summary of missing size imputations, including per-species breakdown."""
        print(f"\n=========== {self.img_dir} ===========")
        print(f"fill_missing_bool: {self.fill_missing_bool}")
        print(f"Total scanned images: {len(self.samples) + self.missing_size_total_count}")
        print(f"   Final dataset size:   {len(self.samples)}")
        print(f"   Missing size found:   {self.missing_size_total_count}")
        print("\nMissing size by species:")
        for species, count in self.missing_size_species.items():
            print(f"      {species}: {count} ")
        
        if not self.fill_missing_bool:
            print(f"Dropped missing:      {self.missing_size_total_count}")
        print("=====================================\n")
