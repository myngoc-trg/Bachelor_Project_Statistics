import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict
from Data_Utility.dataset import PollenFolderWithSizeDataset

class AugmentedPollenDataset(PollenFolderWithSizeDataset):
    """
    Keeps the orginal training dataset intact, and adds extra samples for each minority class so that every class reaches the same size as the largest class (n_max)
    
    For each class c:
        - Keep all orginal n_c samples (no transform change or just base transform with normalization)
        - Add (n_max - n_c) extra samples by re-sampling indices from class c with replacement
        and applying random augmentation to those extra samples only
        
    Biggest class (n_c == n_max): adds 0 extra samples.
    
    Output format stays the same as parent: (img, size, label)
    """
    
    def __init__(self,
        img_dir: str,
        class_to_idx: Dict[str, int],
        size_lookup: Dict[str, Tuple[float, float]],
        species_mean_lookup: Dict[str, Tuple[float, float]],
        fill_missing_bool: bool = False,

        # Transforms:
        transform_base = None,     # always applied  ToTensor + Normalize)
        transform_aug = None,      # ONLY applied to extra samples (PIL augmentations)
        augment: bool = False,     # True if want to do augmentation for extra data
        n_max: int = None,         # if provided, use this as the target max class size instead of computing from data
        global_mean = None
        ,global_std = None
        ,normalize_size = False
        ,quota_bool: bool = False 
        ,print_summary: bool = True
        ,bootstrap_impute_bool: bool = False
        ,species_pool=None
        ,flower_pool=None
        ,seed: int = 42,
        shuffle_final: bool = True
        ,return_flowerid:bool = False
    ):
        """
        NOTES on transforms:
            - transform_aug should be PIL-only random ops (flip/rotate/color jitter, etc.)
            - transform_base should include ToTensor + Normalize 
            - We apply transform_aug only to the extra samples, then base-transform to all samples.
        """
        
        super().__init__(
            img_dir=img_dir,
            class_to_idx=class_to_idx,
            size_lookup=size_lookup,
            species_mean_lookup=species_mean_lookup,
            fill_missing_bool=fill_missing_bool,
            transform=None,               # we will apply transforms ourselves
            print_summary=print_summary,
            return_flowerid=False
        )
        
        self.transform_base = transform_base
        self.transform_aug = transform_aug
        self.augment = augment
        
        self.bootstrap_impute_bool = bootstrap_impute_bool
        self.species_pool = species_pool
        self.flower_pool = flower_pool

        self.normalize_size = normalize_size
        self.quota_bool = quota_bool
        self.size_mean = torch.tensor(global_mean, dtype=torch.float32) if global_mean is not None else None
        self.size_std = torch.tensor(global_std, dtype=torch.float32) if global_std is not None else None

        self.return_flowerid = return_flowerid
                
        per_class_indices = defaultdict(list)
        for i, (_, class_name) in enumerate(self.samples):
            per_class_indices[class_name].append(i)
            
        class_sizes = {c: len(idxs) for c, idxs in per_class_indices.items()}
        n_max = n_max if n_max is not None else max(class_sizes.values()) #get distribution of biggest class
        
        # --- Build index map for the new dataset ---
        # We will store pairs : (base_idx, is_extra)
        # - all originals: (i, False) for i in range(len(self.samples))
        # extras: (some_idx_from_class_c, True) repeated (n_max - n_c) times
        mapping: List[Tuple[int, bool]] = []
        
        # Keep all original samples
        mapping.extend([(i, False) for i in range(len(self.samples))])
                       
        # Add extras per class to reach max
        g = torch.Generator().manual_seed(seed)
        extra_total = 0
        if self.augment:
            for class_name in sorted(per_class_indices.keys()):
                idxs = per_class_indices[class_name]
                n_c = len(idxs)
                deficit = n_max - n_c
                if deficit <= 0:
                    continue
                
                extra_total += deficit
                
                # sample with replacement to reach n_max for every class
                # pick_j - Uniform(0,...,m-1)
                picks = torch.randint(low=0, high=n_c, size=(deficit,), generator=g).tolist()
                for p in picks:
                    mapping.append((idxs[p], True))
                
        if shuffle_final:
            perm = torch.randperm(len(mapping), generator=g).tolist()
            mapping = [mapping[i] for i in perm]

        self.mapping = mapping
        self.n_max = n_max
        self.class_sizes = class_sizes
        self.extra_total = extra_total

        self.final_filenames = [self.filenames[base_idx] for base_idx, _ in self.mapping]
        self.final_flower_ids = [self.flower_ids[base_idx] for base_idx, _ in self.mapping]
        self.final_targets = [self.targets[base_idx] for base_idx, _ in self.mapping]
        self.final_is_extra = [is_extra for _, is_extra in self.mapping]
        
        if print_summary and self.augment:
            print(f"\n=== Augmented-to-max dataset built from '{img_dir}' ===")
            print(f"Original samples kept: {len(self.samples)}")
            print(f"Extra augmented samples added: {self.extra_total}")
            print(f"Final dataset size: {len(self.mapping)}")
            print(f"Target per-class size (n_max): {self.n_max}")
            print("Per-class original sizes:")
            for c in sorted(self.class_sizes.keys()):
                print(f"  {c}: {self.class_sizes[c]} (add {self.n_max - self.class_sizes[c]})")
            print("=====================================================\n")
            
    def __len__(self):
        return len(self.mapping)
    
    def extract_flower_id(self, filename):
        print("DEBUG extract_flower_id got:", type(filename), filename)
        if isinstance(filename, (list, tuple)):
            if len(filename) != 1:
                raise ValueError(f"Expected one filename, got {filename}")
            filename = filename[0]

        if not isinstance(filename, str):
            raise TypeError(f"extract_flower_id expected str, got {type(filename)}: {filename}")

        filename = os.path.basename(filename)
        return filename.split(" ")[-1].split("_")[0]
    
    def __getitem__(self, idx: int):
        base_idx, is_extra = self.mapping[idx]
        img_path, class_name = self.samples[base_idx]
        filename = os.path.basename(img_path)
        
        img = Image.open(img_path).convert('RGB')
        
        if is_extra and self.transform_aug is not None:
            img = self.transform_aug(img)
        
        if self.transform_base is not None:
            img = self.transform_base(img)
            
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
                else:
                    major, minor = sampled

            else:
                # fallback to species mean
                if class_name in self.species_mean_lookup:
                    if self.quota_bool:
                        major, minor, quota = self.species_mean_lookup[class_name]
                    else:
                        major, minor = self.species_mean_lookup[class_name]
                else:
                    raise ValueError(
                        f"Size data missing for image '{filename}' and no species mean available for '{class_name}'.")
        
        if self.quota_bool:
            size = torch.tensor([major, minor, quota], dtype=torch.float32)
        else:
            size = torch.tensor([major, minor], dtype=torch.float32)
            
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)   
        
         
        if self.normalize_size and (self.size_mean is not None) and (self.size_std is not None):
            size = (size - self.size_mean) / torch.clamp(self.size_std, min=1e-8)
        
        if self.return_flowerid:
            flower_id = self.final_flower_ids[idx]
            return img, size, label, filename, flower_id 

        return img, size, label
              