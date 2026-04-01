import os
import re
from typing import Dict, Tuple, List
import pandas as pd


def lookup_size_from_excel(excel_path: str, stats_filenames=None, quota_bool: bool = False):
    """
    Reads an Excel file containing image names and their corresponding sizes,
    and returns a dictionary mapping image names to their sizes. and quota information.

    Args:
        excel_path (str): The path to the Excel file.
        stats_filenames (List[str], optional): A list of filenames to include in statistics.
        quota_bool (bool): Whether to include quota information.

    Returns:
        Dict[str, Tuple[int, int]]: A dictionary where keys are image names and values are tuples of (width, height).
    """
    df = pd.read_excel(excel_path)
    required_columns = {'anyname', 'ping_name', 'majoraxis', 'minoraxis'}
    lookup = {}
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {df.columns.tolist()}")
        
    # Fill missing values with species /anyname mean majoraxis and minoraxis
    # --- Tracking how many missing values we have before filling. In total:
    df['is_missing'] = df['majoraxis'].isna() | df['minoraxis'].isna()
    missing_count = df['is_missing'].sum()
    
    
    # -- Tracking missing per species
    missing_per_species = (df.groupby('anyname')['is_missing'].sum().sort_values(ascending=False))
    print(f"Total samples: {len(df)}")
    if missing_count > 0:
        print("\n======= Missing size Statistics =======")
        print(f"Found {missing_count} rows with missing size data. Filling with species means where possible.")
        print("\nMissing size count per species:")
        for species, count in missing_per_species.items():
            if count > 0:
                print(f"  {species}: {count}")
        print("======================================\n")
    else:
        print("No missing size data found. All rows have valid majoraxis and minoraxis values in Excel file.")
        
    # Convert size columns to numeric (missing -> NaN)
    df['majoraxis'] = pd.to_numeric(df['majoraxis'], errors='coerce')
    df['minoraxis'] = pd.to_numeric(df['minoraxis'], errors='coerce')
    
    # quota information
    if quota_bool:
        df['majoraxis'] = df['majoraxis'].clip(lower=1e-8)  # Ensure no zero division
        df['quota'] = df['minoraxis'] / df['majoraxis']
        feature_vec = ['majoraxis', 'minoraxis', 'quota']
    else:
        feature_vec = ['majoraxis', 'minoraxis']
    
    species_means = df.groupby('anyname')[feature_vec].mean()
    
    for idx, row in df.iterrows():
        if pd.isna(row['majoraxis']) or pd.isna(row['minoraxis']):
            species = row['anyname']
            if species in species_means.index:
                row['majoraxis'] = species_means.loc[species, 'majoraxis']
                row['minoraxis'] = species_means.loc[species, 'minoraxis']
            else:
                raise ValueError(f"Missing size data for species '{species}' and no mean available.")
    
    species_mean_lookup = {}
    for species, row in species_means.iterrows():
        species_mean_lookup[species] = tuple(float(row[col]) for col in feature_vec)
      

    if stats_filenames is not None:
        stats_df = df[df["ping_name"].astype(str).isin(stats_filenames)].copy()
        if len(stats_df) == 0:
            raise ValueError(
                "stats_filenames produced 0 matching rows in Excel. "
                "Check filename matching between TRAIN_DIR and Excel ping_name."
            )
    else:
        stats_df = df

    global_mean = tuple(float(stats_df[col].mean()) for col in feature_vec)

    global_std = tuple(float(stats_df[col].std(ddof=0)) for col in feature_vec)

    # Original size available, fill lookup
    for _, row in df.iterrows():
        image_name = row['ping_name']
        majoraxis = float(row['majoraxis'])
        minoraxis = float(row['minoraxis'])
        lookup[image_name] = tuple(float(row[col]) for col in feature_vec)
    
    
        
    return lookup, species_mean_lookup, global_mean, global_std











