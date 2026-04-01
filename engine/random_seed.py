import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For stricter reproducibility on GPU
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Raise an error if a nondeterministic op is used
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int):
    """
    Ensure each DataLoader worker has a reproducible seed.
    Important when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)