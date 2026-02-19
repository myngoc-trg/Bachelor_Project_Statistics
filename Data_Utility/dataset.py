from email.policy import default
import os
from sklearn.base import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict

class PollenFolderWithSizeDataset(Dataset):
    def __init__(self, img_dir: str, class_to_idx: Dict[str, int], size_lookup: Dict[str, Tuple[float, float]], species_mean_lookup: Dict[str, Tuple[float, float]], global_mean: Tuple[float, float], transform=None):
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
        self.global_mean = global_mean
        self.transform = transform
        
        # Build list of (image_path, label) tuples
        self.samples = self._scan_root()
        
        # Count missing size entries for reporting
        self.missing_size_count = 0
        self.missing_filled_global_count = 0
        self.missing_filled_species_count = 0
        
        # Per-species missing count for detailed reporting
        self.species_missing_count = defaultdict(int)
        self.species_filled_species_mean = defaultdict(int)
        self.species_filled_global_mean = defaultdict(int)
        
        self._seen_indices = set()  # To track which indices have been processed for summary printing
        self._summary_printed = False  # Flag to ensure summary is printed only once

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
            self.missing_size_count += 1
            self.species_missing_count[class_name] += 1
            
            if class_name not in self.species_mean_lookup:
                # last resort: fill with overall mean
                #print(f"\nWarning: No species mean available for '{class_name}'. Filling with global mean.")
                self.missing_filled_global_count += 1
                self.species_filled_global_mean[class_name] += 1
                major, minor = self.global_mean
                
            else:
                #print(f"\nFilling missing size for '{filename}' with species mean for '{class_name}'.")
                self.missing_filled_species_count += 1
                self.species_filled_species_mean[class_name] += 1
                major, minor = self.species_mean_lookup[class_name]
           
        size = torch.tensor([major, minor], dtype=torch.float32)
        
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        
        # Print missing summary after full dataset has been processed
        if (not self._summary_printed) and (len(self._seen_indices) == len(self.samples)):
            print(f"\n\n=========== {self.img_dir} - Finished processing dataset. ==================")
            if self.missing_size_count > 0:
                self.print_missing_summary()
                self._summary_printed = True
            else:
                print(f"\nNo missing size data found in the image dataset of {self.img_dir}.")
           
        
        return img, size, label
    
    def print_missing_summary(self, top_k: int | None = None):
        """Print summary of missing size imputations, including per-species breakdown."""
        total = len(self.samples)

        print("\n========== Dataset Missing Size Summary ==========")
        print(f"Total samples: {total}")
        print(f"Missing size entries: {self.missing_size_count}")
        print(f"Filled with species mean: {self.missing_filled_species_count}")
        print(f"Filled with global mean:  {self.missing_filled_global_count}")
        print("=================================================")

        if self.missing_size_count == 0:
            return

        # Build rows
        rows = []
        for species in self.species_missing_count.keys():
            miss = self.species_missing_count[species]
            sp = self.species_filled_species_mean[species]
            gl = self.species_filled_global_mean[species]
            rows.append((species, miss, sp, gl))

        # Sort by missing (desc), then species name
        rows.sort(key=lambda x: (-x[1], x[0]))

        if top_k is not None:
            rows = rows[:top_k]

        # Column widths
        name_w = max(len("Species"), max((len(r[0]) for r in rows), default=7))
        miss_w = max(len("Missing"), max((len(str(r[1])) for r in rows), default=7))
        sp_w   = max(len("SpeciesMean"), max((len(str(r[2])) for r in rows), default=11))
        gl_w   = max(len("GlobalMean"), max((len(str(r[3])) for r in rows), default=10))

        # Header
        print("\nPer-species missing size summary")
        print(
            f"{'Species'.ljust(name_w)}  "
            f"{'Missing'.rjust(miss_w)}  "
            f"{'SpeciesMean'.rjust(sp_w)}  "
            f"{'GlobalMean'.rjust(gl_w)}"
        )
        print("-" * (name_w + miss_w + sp_w + gl_w + 6))

        # Rows
        for species, miss, sp, gl in rows:
            print(
                f"{species.ljust(name_w)}  "
                f"{str(miss).rjust(miss_w)}  "
                f"{str(sp).rjust(sp_w)}  "
                f"{str(gl).rjust(gl_w)}"
            )


        total_miss_by_species = sum(r[1] for r in rows)
        print(f"\n(Shown missing total: {total_miss_by_species} — dataset missing total: {self.missing_size_count})")
