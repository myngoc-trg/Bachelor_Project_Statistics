from email.policy import default
from fileinput import filename
import os
from sklearn.base import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict

class PollenFolderWithSizeDataset(Dataset):
    def __init__(self, img_dir: str, class_to_idx: Dict[str, int], 
                 size_lookup: Dict[str, Tuple[float, float]], 
                 species_mean_lookup: Dict[str, Tuple[float, float]], 
                 fill_missing_bool: bool = False, 
                 transform=None,
                 global_mean = None
                 ,global_std = None
                 ,normalize_size = False 
                 ,print_summary: bool = True
                 ,quota_bool: bool = False
                 ,bootstrap_impute_bool: bool = False
                 ,species_pool=None
                 ,flower_pool=None
                 ):
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
        self.quota_bool = quota_bool
        
        self.bootstrap_impute_bool = bootstrap_impute_bool
        self.species_pool = species_pool  # expected name for bootstrap resolver
        self.species_bool = species_pool  # legacy alias (if referenced elsewhere)
        self.flower_pool = flower_pool

        self.normalize_size = normalize_size
        self.size_mean = torch.tensor(global_mean, dtype=torch.float32) if global_mean is not None else None
        self.size_std = torch.tensor(global_std, dtype=torch.float32) if global_std is not None else None
        
        # Build list of (image_path, label) tuples
        all_samples = self._scan_root()
        
        # Count missing size entries for reporting
        self.missing_size_total_count = 0
        
        # Per-species missing count for detailed reporting
        self.missing_size_species = defaultdict(int)
        
        self._summary_printed = False  # Flag to ensure summary is printed only once
        
        self.samples = [] # list of (img_path, label)
        self.targets = [] # list of class_name
        
        for img_path, class_name in all_samples:
            filename = os.path.basename(img_path)
            if filename in self.size_lookup:
                self.samples.append((img_path, class_name))
                self.targets.append(class_name)
                continue
            
            self.missing_size_total_count += 1
            self.missing_size_species[class_name] += 1
            
            if not self.fill_missing_bool:
                continue
            
            self.samples.append((img_path, class_name)) # keep the sample even if size is missing, will handle in __getitem__ with filling logic
            self.targets.append(class_name)
        
        if print_summary:
            print(f"\nInitialized dataset from '{self.img_dir}' with {len(self.samples)} samples.")
            if self.missing_size_total_count > 0:
                self.print_missing_summary()
            else:
                print(f"\nNo missing size data found in the image dataset of {self.img_dir}.")
        
    
    def __len__(self):
        return len(self.samples)
    
    def get_class_counts(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Returns counts per class index (0..K-1) WITHOUT loading images.
        Uses self.targets (strings) built at init time.

        Output shape: [K], dtype long
        """
        K = 10  # Assuming 10 classes, can be made dynamic if needed
        counts = torch.zeros(K, dtype=torch.long)

        for class_name in self.targets:
            idx = self.class_to_idx[class_name]
            counts[idx] += 1

        return counts.to(device) if device is not None else counts

    def get_class_probs(self, device: torch.device | None = None) -> torch.Tensor:
        """Convenience: class probabilities from counts."""
        counts = self.get_class_counts(device=device).float()
        return counts / counts.sum().clamp(min=1.0)
    
    def get_effective_num_weights(self, beta: float = 0.999, device=None) -> torch.Tensor:
        """
        Compute class weights using the Effective Number of Samples formula:
            E_c = (1 - beta^n_c) / (1 - beta)
            w_c = 1 / E_c

        We normalize weights so mean weight = 1.
        """

        counts = self.get_class_counts().float()

        # Avoid division by zero
        counts = counts.clamp(min=1.0)

        effective_num = (1.0 - beta ** counts) / (1.0 - beta)
        weights = 1.0 / effective_num

        # Normalize (important for stability)
        weights = weights / weights.mean()

        if device is not None:
            weights = weights.to(device)

        return weights
    
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
    
    def extract_flower_id(self, filename: str) -> str:
        return filename.split(" ")[-1].split("_")[0]
    
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
            if self.quota_bool:
                major, minor, quota = self.size_lookup[filename]
            else:
                major, minor = self.size_lookup[filename]
        
        else:
            if self.bootstrap_impute_bool:
                from Data_Utility.bootstrap_impute import sample_bootstrap_size
                #print(f"Bootstrapping size for missing image '{filename}' in class '{class_name}'...")
                flower_id = self.extract_flower_id(filename)

                sampled = sample_bootstrap_size(
                    species=class_name,
                    flower_id=flower_id,
                    species_pool=self.species_pool,
                    flower_pool=self.flower_pool,
                    quota_bool=self.quota_bool
                )

                if self.quota_bool:
                    major, minor, quota = sampled
                    #print(f"Bootstrapped size for '{filename} in class '{class_name}: major={major}, minor={minor}, quota={quota}")
                else:
                    major, minor = sampled
                    #print(f"Bootstrapped size for '{filename}'in class '{class_name}: major={major}, minor={minor}")

            else:
                # fallback to species mean
                if class_name in self.species_mean_lookup:
                    #print(f"Using species mean for missing image '{filename}' in class '{class_name}'...")
                    if self.quota_bool:
                        #print(f"Using species mean for '{filename}' in class '{class_name}': major={self.species_mean_lookup[class_name][0]}, minor={self.species_mean_lookup[class_name][1]}, quota={self.species_mean_lookup[class_name][2]}")
                        major, minor, quota = self.species_mean_lookup[class_name]
                    else:
                        #print(f"Using species mean for '{filename}' in class '{class_name}': major={self.species_mean_lookup[class_name][0]}, minor={self.species_mean_lookup[class_name][1]}")
                        major, minor = self.species_mean_lookup[class_name]
                else:
                    raise ValueError(
                        f"Size data missing for image '{filename}' and no species mean available for '{class_name}'."
                    )        
                
       
        if self.quota_bool:
            size = torch.tensor([major, minor, quota], dtype=torch.float32)
        else:
            size = torch.tensor([major, minor], dtype=torch.float32)
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)   
        
        if self.normalize_size and (self.size_mean is not None) and (self.size_std is not None):
            size = (size - self.size_mean) / torch.clamp(self.size_std, min=1e-8)

        return img, size, label
    
    def print_missing_summary(self, top_k: int | None = None):
        """Print summary of missing size imputations, including per-species breakdown."""
        print(f"\n=========== {self.img_dir} ===========")
        print(f"fill_missing_bool: {self.fill_missing_bool}")
        print(f"bootstrap_impute_bool: {self.bootstrap_impute_bool}")
        print(f"Total scanned images: {len(self.samples) + self.missing_size_total_count}")
        print(f"   Final dataset size:   {len(self.samples)}")
        print(f"   Missing size found:   {self.missing_size_total_count}")
        print("\nMissing size by species:")
        for species, count in self.missing_size_species.items():
            print(f"      {species}: {count} ")
        
        if not self.fill_missing_bool:
            print(f"Dropped missing:      {self.missing_size_total_count}")
        print("=====================================\n")
