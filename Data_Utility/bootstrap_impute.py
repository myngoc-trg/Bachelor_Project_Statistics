import pandas as pd
from collections import defaultdict
import os
import random


def extract_flower_id(filename: str) -> str:
    """
    Example:
    'Crepis capillaris 1000322_135.png' -> '1000322'
    """
    return filename.split(" ")[-1].split("_")[0]

def build_bootstrap_pools(excel_path: str, train_filenames: list[str], quota_bool: bool = False):
    df = pd.read_excel(excel_path)

    df = df[df["ping_name"].astype(str).isin(train_filenames)].copy()

    df["majoraxis"] = pd.to_numeric(df["majoraxis"], errors="coerce")
    df["minoraxis"] = pd.to_numeric(df["minoraxis"], errors="coerce")
    df = df.dropna(subset=["majoraxis", "minoraxis"]).copy()

    df["flower_id"] = df["ping_name"].astype(str).apply(extract_flower_id)

    if quota_bool:
        df["majoraxis"] = df["majoraxis"].clip(lower=1e-8)
        df["quota"] = df["minoraxis"] / df["majoraxis"]

    species_pool = defaultdict(list)
    flower_pool = defaultdict(list)

    for _, row in df.iterrows():
        species = row["anyname"]
        flower_id = row["flower_id"]

        if quota_bool:
            pair = (
                float(row["majoraxis"]),
                float(row["minoraxis"]),
                float(row["quota"])
            )
        else:
            pair = (
                float(row["majoraxis"]),
                float(row["minoraxis"])
            )

        species_pool[species].append(pair)
        flower_pool[(species, flower_id)].append(pair)

    return species_pool, flower_pool

def sample_bootstrap_size(
    species: str,
    flower_id: str,
    species_pool,
    flower_pool
    ,quota_bool: bool =False
    ,rng=None
):
    """
    First try same species + same flower.
    If not available, fall back to same species.
    
    Returns a (major, minor) pair, or (major, minor, quota) if quota_bool is True.
    """
    if rng is None:
        rng = random

    if species_pool is None or flower_pool is None:
        raise ValueError("Bootstrap pools must be provided for imputation.")
    
    if (species, flower_id) in flower_pool and len(flower_pool[(species, flower_id)]) > 0:
        sampled = rng.choice(flower_pool[(species, flower_id)])

    elif species in species_pool and len(species_pool[species]) > 0:
        sampled = rng.choice(species_pool[species])

    else:
        raise ValueError(f"No bootstrap donor available for species '{species}'.")
    
    
    if quota_bool:
        if len((sampled)) == 3:
            #print(f"quota_bool={quota_bool} | Sampled for species '{species}', flower_id '{flower_id}': {sampled}")
            return sampled
        elif len(sampled) == 2:
            #print(f"quota_bool={quota_bool} | Sampled for species '{species}', flower_id '{flower_id}': {sampled}")
            major, minor = sampled
            quota = minor / major if major > 1e-8 else 0.0
            return (major, minor, quota)
    else:
        if len(sampled) == 2:
            #print(f"quota_bool={quota_bool} | Sampled for species '{species}', flower_id '{flower_id}': {sampled}")
            return sampled
        elif len(sampled) == 3:
            #print(f"quota_bool={quota_bool} | Sampled for species '{species}', flower_id '{flower_id}': {sampled}")
            major, minor, _ = sampled
            return (major, minor)
            
    
    
    return sampled
        