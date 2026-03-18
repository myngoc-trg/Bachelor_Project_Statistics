import os
import torch
import torchvision.transforms as T
from torch.utils.data import random_split
from Data_Utility.split_groups import group_train_val_split

from Data_Utility.lookup_size import lookup_size_from_excel
from Data_Utility.dataset import PollenFolderWithSizeDataset
from Data_Utility.augmented_dataset import AugmentedPollenDataset
from Data_Utility.config import TRAIN_DIR, TEST_DIR, EXCEL_PATH

def list_image_filenames(root_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    names = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fn in os.listdir(class_dir):
            if fn.lower().endswith(exts):
                names.append(fn) 
    return set(names)


def setup_datasets(
    val_bool=False,
    aug_bool=False,
    norm_bool=False,
    sep_val_bool=False,
    seed=42
):
    """
    Builds datasets using fixed project paths.
    Optional:
        - validation split
        - augmentation
        - Normalization of size
    """
    train_filenames = list_image_filenames(TRAIN_DIR)

    size_lookup, species_mean_lookup, train_mean, train_std = lookup_size_from_excel(
    EXCEL_PATH,
    stats_filenames=train_filenames
    )
    #size_lookup, species_mean_lookup, global_mean, global_std = lookup_size_from_excel(EXCEL_PATH)

    classes = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    base_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    aug_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=20),
        T.RandomApply([
            T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            )
        ], p=0.4),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.15),
    ])

    val_dataset = None

    if aug_bool:

        train_dataset = AugmentedPollenDataset(
            img_dir=TRAIN_DIR,
            class_to_idx=class_to_idx,
            size_lookup=size_lookup,
            species_mean_lookup=species_mean_lookup,
            fill_missing_bool=True,
            transform_base=base_transform,
            transform_aug=aug_transform,
            augment=True,
            normalize_size=norm_bool,
            global_mean=train_mean,
            global_std=train_std,
            seed=seed,
            shuffle_final=True,
            print_summary=True
        )

        test_dataset = AugmentedPollenDataset(
            img_dir=TEST_DIR,
            class_to_idx=class_to_idx,
            size_lookup=size_lookup,
            species_mean_lookup=species_mean_lookup,
            fill_missing_bool=False,
            transform_base=base_transform,
            transform_aug=None,
            augment=False,
            normalize_size=norm_bool,
            global_mean=train_mean,
            global_std=train_std,
            seed=seed,
            shuffle_final=True,
            print_summary=True
        )

    else:
        train_dataset = PollenFolderWithSizeDataset(
            img_dir=TRAIN_DIR,
            class_to_idx=class_to_idx,
            size_lookup=size_lookup,
            species_mean_lookup=species_mean_lookup,
            transform=base_transform
            ,fill_missing_bool=True
            ,normalize_size=norm_bool,
            global_mean=train_mean,
            global_std=train_std
        )
        
        test_dataset = PollenFolderWithSizeDataset(
            img_dir=TEST_DIR,
            class_to_idx=class_to_idx,
            size_lookup=size_lookup,
            species_mean_lookup=species_mean_lookup,
            transform=base_transform
            ,fill_missing_bool=False
            ,normalize_size=norm_bool,
            global_mean=train_mean,
            global_std=train_std
        )


    if val_bool:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=generator
        )

        print(f"Train subset size: {len(train_dataset)}")
        print(f"Validation subset size: {len(val_dataset)}")
        
    elif sep_val_bool:
        train_dataset, val_dataset = group_train_val_split(
            train_dataset,
            val_ratio=0.2,
            seed=seed
        )
        
        

    print(f"Train subset size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation subset size: {len(val_dataset)}")

    print("Dataset setup complete.")

    return train_dataset, val_dataset, test_dataset, class_to_idx, idx_to_class, classes