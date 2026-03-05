"""Step 2: Load formation mask polygons from CSV files."""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import MASKS_DIR, MASK_FILES, OUTPUT_DIR


def load_all_masks():
    """Load all formation mask CSVs into a dict of DataFrames."""
    masks = {}
    for name, filename in MASK_FILES.items():
        path = os.path.join(MASKS_DIR, filename)
        df = pd.read_csv(path)
        masks[name] = df
        print(f"  {name:<10} {len(df):>4} vertices  "
              f"X:[{df['X'].min():.0f}, {df['X'].max():.0f}]  "
              f"Y:[{df['Y'].min():.0f}, {df['Y'].max():.0f}]")
    return masks


def save_masks(masks):
    """Save combined mask data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    for name, df in masks.items():
        df = df.copy()
        df["formation"] = name
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "formation_masks.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined masks ({len(combined)} vertices) to: {out_path}")
    return combined


if __name__ == "__main__":
    print("Loading formation masks...")
    masks = load_all_masks()
    save_masks(masks)
