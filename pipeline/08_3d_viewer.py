"""Generate data for the interactive 3D WebGL viewer.

Exports subsampled formation meshes as JSON for Three.js rendering,
plus drill hole locations.
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, OUTPUT_DIR
)

MODEL_DIR = os.path.join(OUTPUT_DIR, "neural_field")
WEB_DIR = os.path.join(OUTPUT_DIR, "web")
VIEWER_DIR = os.path.join(WEB_DIR, "viewer")
os.makedirs(VIEWER_DIR, exist_ok=True)


def rgb_to_hex_int(rgb):
    return (int(rgb[0]*255) << 16) | (int(rgb[1]*255) << 8) | int(rgb[2]*255)


def export_formation_cubes(volume, step=6):
    """Export subsampled cube positions per formation as binary buffers."""
    print(f"Exporting formations (step={step})...")

    # Subsample
    vol_sub = volume[::step, ::step, ::step]
    nx_s, ny_s, nz_s = vol_sub.shape

    formations = {}
    for code, name in LITHOLOGY_MAP.items():
        mask = vol_sub == code
        if mask.sum() == 0:
            continue

        indices = np.argwhere(mask)
        # Convert to real-world coords, then normalize for Three.js scene
        coords = np.empty((len(indices), 3), dtype=np.float32)
        coords[:, 0] = ORIGIN[0] + indices[:, 0] * step * SPACING[0]
        coords[:, 1] = ORIGIN[1] + indices[:, 1] * step * SPACING[1]
        coords[:, 2] = ORIGIN[2] + indices[:, 2] * step * SPACING[2]

        formations[name] = {
            "code": code,
            "count": int(len(indices)),
            "color": rgb_to_hex_int(LITHOLOGY_COLORS[code]),
            "color_rgb": [int(c*255) for c in LITHOLOGY_COLORS[code]],
        }

        # Save as binary float32 (x,y,z interleaved)
        bin_path = os.path.join(VIEWER_DIR, f"{name}.bin")
        coords.tofile(bin_path)
        print(f"  {name}: {len(indices):,} cubes -> {bin_path}")

    return formations


def export_prediction_cubes(pred_volume, gt_volume, step=6):
    """Export predicted cubes that differ from ground truth."""
    print(f"Exporting prediction differences (step={step})...")

    gt_sub = gt_volume[::step, ::step, ::step]
    pred_sub = pred_volume[::step, ::step, ::step]

    # Only where prediction differs AND ground truth is valid
    diff_mask = (gt_sub != NODATA) & (gt_sub != pred_sub)
    correct_mask = (gt_sub != NODATA) & (gt_sub == pred_sub)

    diff_indices = np.argwhere(diff_mask)
    correct_indices = np.argwhere(correct_mask)

    # For diff points, save position + gt_class + pred_class
    diff_data = np.empty((len(diff_indices), 5), dtype=np.float32)
    diff_data[:, 0] = ORIGIN[0] + diff_indices[:, 0] * step * SPACING[0]
    diff_data[:, 1] = ORIGIN[1] + diff_indices[:, 1] * step * SPACING[1]
    diff_data[:, 2] = ORIGIN[2] + diff_indices[:, 2] * step * SPACING[2]
    diff_data[:, 3] = gt_sub[diff_mask]
    diff_data[:, 4] = pred_sub[diff_mask]

    diff_path = os.path.join(VIEWER_DIR, "errors.bin")
    diff_data.tofile(diff_path)
    print(f"  Errors: {len(diff_indices):,} cubes")
    print(f"  Correct: {len(correct_indices):,} cubes")

    return len(diff_indices), len(correct_indices)


def export_drill_holes(split_info):
    """Export drill hole locations as line segments."""
    print("Exporting drill holes...")
    wells = []
    radius = split_info["well_radius_cells"]

    z_top = ORIGIN[2]
    z_bottom = ORIGIN[2] + (NZ - 1) * SPACING[2]

    for wx, wy in zip(split_info["well_x"], split_info["well_y"]):
        wells.append({
            "x": wx, "y": wy,
            "z_top": z_top, "z_bottom": z_bottom,
            "radius_m": radius * abs(SPACING[0]),
        })

    return wells


def export_metadata(formations, wells, n_errors, n_correct):
    """Save all metadata as JSON for the viewer."""
    # Scene bounds for camera setup
    x_min = ORIGIN[0]
    x_max = ORIGIN[0] + (NX - 1) * SPACING[0]
    y_min = ORIGIN[1] + (NY - 1) * SPACING[1]
    y_max = ORIGIN[1]
    z_min = ORIGIN[2] + (NZ - 1) * SPACING[2]
    z_max = ORIGIN[2]

    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    extent = [x_max - x_min, y_max - y_min, z_max - z_min]

    meta = {
        "formations": formations,
        "wells": wells,
        "bounds": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_min, z_max],
            "center": center,
            "extent": extent,
        },
        "errors": {"n_errors": n_errors, "n_correct": n_correct},
        "grid": {"nx": NX, "ny": NY, "nz": NZ,
                 "spacing": SPACING.tolist(), "origin": ORIGIN.tolist()},
    }

    meta_path = os.path.join(VIEWER_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    return meta


if __name__ == "__main__":
    gt = np.load(os.path.join(OUTPUT_DIR, "voxet_lithology.npy"))
    pred = np.load(os.path.join(MODEL_DIR, "predicted_volume.npy"))

    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split_info = json.load(f)

    formations = export_formation_cubes(gt, step=6)
    n_err, n_cor = export_prediction_cubes(pred, gt, step=6)
    wells = export_drill_holes(split_info)
    export_metadata(formations, wells, n_err, n_cor)

    print("\nData export complete. Now building HTML viewer...")
