import os
import pandas as pd
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit


def group_train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Splits dataset into train/val ensuring flower_id does not overlap.
    Assumes dataset.samples = [(img_path, class_name), ...]
    """

    # Extract filenames from dataset
    filepaths = [p for (p, _) in dataset.samples]
    filenames = [os.path.basename(p) for p in filepaths]

    df = pd.DataFrame({'filename': filenames})

    # Extract flower_id 
    df['flower_id'] = (
        df['filename']
        .str.split(' ').str[-1]
        .str.split('_').str[0]
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )

    train_idx, val_idx = next(
        splitter.split(df['filename'], groups=df['flower_id'])
    )

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    return train_subset, val_subset