"""Step 4: Export the 3D model to various formats."""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, OUTPUT_DIR
)


def load_volume():
    path = os.path.join(OUTPUT_DIR, "voxet_lithology.npy")
    return np.load(path)


def export_vti(volume):
    """Export as VTK ImageData (.vti) for ParaView."""
    import pyvista as pv

    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.origin = (ORIGIN[0], ORIGIN[1] + (NY - 1) * SPACING[1], ORIGIN[2] + (NZ - 1) * SPACING[2])
    grid.spacing = (abs(SPACING[0]), abs(SPACING[1]), abs(SPACING[2]))

    vol_flipped = volume[:, ::-1, ::-1].copy()
    lith = vol_flipped.flatten(order='F').astype(np.float32)
    lith[lith == NODATA] = np.nan
    grid.cell_data["lithology"] = lith

    out = os.path.join(OUTPUT_DIR, "adavale_lithology.vti")
    grid.save(out)
    print(f"Saved VTI: {out} ({os.path.getsize(out)/1e6:.0f} MB)")
    return grid


def export_per_formation_stl(volume):
    """Export each formation as a separate STL surface mesh."""
    import pyvista as pv

    stl_dir = os.path.join(OUTPUT_DIR, "stl")
    os.makedirs(stl_dir, exist_ok=True)

    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.origin = (ORIGIN[0], ORIGIN[1] + (NY - 1) * SPACING[1], ORIGIN[2] + (NZ - 1) * SPACING[2])
    grid.spacing = (abs(SPACING[0]), abs(SPACING[1]), abs(SPACING[2]))

    vol_flipped = volume[:, ::-1, ::-1].copy()
    lith = vol_flipped.flatten(order='F').astype(np.float32)
    lith[lith == NODATA] = np.nan
    grid.cell_data["lithology"] = lith

    for code, name in LITHOLOGY_MAP.items():
        try:
            formation = grid.threshold(value=(code - 0.5, code + 0.5), scalars="lithology")
            if formation.n_cells == 0:
                print(f"  {name}: no cells, skipping")
                continue
            surface = formation.extract_surface()
            out = os.path.join(stl_dir, f"{name}.stl")
            surface.save(out)
            print(f"  {name}: {surface.n_cells:,} faces -> {out}")
        except Exception as e:
            print(f"  {name}: failed ({e})")

    print(f"\nSTL files saved to: {stl_dir}")


def export_per_formation_obj(volume):
    """Export each formation as OBJ."""
    import pyvista as pv

    obj_dir = os.path.join(OUTPUT_DIR, "obj")
    os.makedirs(obj_dir, exist_ok=True)

    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.origin = (ORIGIN[0], ORIGIN[1] + (NY - 1) * SPACING[1], ORIGIN[2] + (NZ - 1) * SPACING[2])
    grid.spacing = (abs(SPACING[0]), abs(SPACING[1]), abs(SPACING[2]))

    vol_flipped = volume[:, ::-1, ::-1].copy()
    lith = vol_flipped.flatten(order='F').astype(np.float32)
    lith[lith == NODATA] = np.nan
    grid.cell_data["lithology"] = lith

    for code, name in LITHOLOGY_MAP.items():
        try:
            formation = grid.threshold(value=(code - 0.5, code + 0.5), scalars="lithology")
            if formation.n_cells == 0:
                continue
            surface = formation.extract_surface()
            out = os.path.join(obj_dir, f"{name}.obj")
            pv.save_meshio(out, surface)
            print(f"  {name}: {surface.n_faces:,} faces -> {out}")
        except Exception as e:
            print(f"  {name}: failed ({e})")

    print(f"\nOBJ files saved to: {obj_dir}")


def export_numpy_per_formation(volume):
    """Export binary masks per formation as .npy files (useful for ML)."""
    npy_dir = os.path.join(OUTPUT_DIR, "npy_masks")
    os.makedirs(npy_dir, exist_ok=True)

    for code, name in LITHOLOGY_MAP.items():
        mask = (volume == code).astype(np.uint8)
        count = mask.sum()
        if count == 0:
            continue
        out = os.path.join(npy_dir, f"{name}_mask.npy")
        np.save(out, mask)
        print(f"  {name}: {count:,} voxels -> {out}")

    # Also save the integer label volume (no nodata, replaced with 0)
    labels = volume.copy()
    labels[labels == NODATA] = 0
    out = os.path.join(npy_dir, "labels.npy")
    np.save(out, labels.astype(np.int8))
    print(f"\n  Label volume (int8) -> {out}")
    print(f"\nNumpy masks saved to: {npy_dir}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volume = load_volume()
    print(f"Volume loaded: {volume.shape}\n")

    print("=== Exporting VTI ===")
    export_vti(volume)

    print("\n=== Exporting per-formation STL ===")
    export_per_formation_stl(volume)

    print("\n=== Exporting per-formation numpy masks (for ML) ===")
    export_numpy_per_formation(volume)

    print("\nAll exports complete!")
