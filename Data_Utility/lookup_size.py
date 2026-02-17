import os
import re
from typing import Dict, Tuple, List
import pandas as pd

def lookup_size_from_excel(excel_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Reads an Excel file containing image names and their corresponding sizes,
    and returns a dictionary mapping image names to their sizes.

    Args:
        excel_path (str): The path to the Excel file.

    Returns:
        Dict[str, Tuple[int, int]]: A dictionary where keys are image names and values are tuples of (width, height).
    """
    df = pd.read_excel(excel_path)
    required_columns = {'ping_name', 'majoraxis', 'minoraxis'}
    
    lookup = {}
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {df.columns.tolist()}")
    
    for _, row in df.iterrows():
        image_name = row['ping_name']
        majoraxis = float(row['majoraxis'])
        minoraxis = float(row['minoraxis'])
        lookup[image_name] = (majoraxis, minoraxis)
    
    return lookup