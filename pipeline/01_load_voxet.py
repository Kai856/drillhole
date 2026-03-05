"""Step 1: Load the GoCAD voxet binary and inspect lithology classes."""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    VOXET_FILE, NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, OUTPUT_DIR
)


def load_voxet():
    """Load the .vop1 raw binary file into a numpy array."""
    print(f"Loading voxet: {os.path.basename(VOXET_FILE)}")
    print(f"Expected shape: ({NX}, {NY}, {NZ}) = {NX*NY*NZ:,} voxels")

    data = np.fromfile(VOXET_FILE, dtype=">f4")  # big-endian float32
    print(f"Loaded {data.size:,} values ({data.nbytes / 1e6:.0f} MB)")

    if data.size != NX * NY * NZ:
        print(f"WARNING: expected {NX*NY*NZ:,} but got {data.size:,}")
        return None

    # GOCAD voxets: X varies fastest (Fortran/column-major order)
    volume = data.reshape((NX, NY, NZ), order='F')
    return volume


def inspect_volume(volume):
    """Print lithology class statistics."""
    print(f"\nVolume shape: {volume.shape}")
    print(f"Value range: [{volume.min():.0f}, {volume.max():.0f}]")

    nodata_count = np.sum(volume == NODATA)
    valid_count = volume.size - nodata_count
    print(f"No-data voxels: {nodata_count:,} ({nodata_count/volume.size*100:.1f}%)")
    print(f"Valid voxels:   {valid_count:,} ({valid_count/volume.size*100:.1f}%)")

    valid_mask = volume != NODATA
    unique, counts = np.unique(volume[valid_mask], return_counts=True)

    print(f"\nLithology class distribution (valid voxels only):")
    print(f"{'Code':>6} {'Formation':<12} {'Count':>12} {'Percent':>8}")
    print("-" * 42)
    for u, c in zip(unique, counts):
        code = int(u)
        name = LITHOLOGY_MAP.get(code, "UNKNOWN")
        print(f"{code:>6} {name:<12} {c:>12,} {c/valid_count*100:>7.1f}%")


def save_numpy(volume):
    """Save as .npy for fast reloading."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "voxet_lithology.npy")
    np.save(out_path, volume)
    print(f"\nSaved numpy array to: {out_path}")
    return out_path


if __name__ == "__main__":
    volume = load_voxet()
    if volume is not None:
        inspect_volume(volume)
        save_numpy(volume)
