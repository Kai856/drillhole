"""Step 3: Visualize the 3D lithology model with PyVista and matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, OUTPUT_DIR, MASKS_DIR, MASK_FILES
)


def load_volume():
    """Load the saved numpy voxet."""
    path = os.path.join(OUTPUT_DIR, "voxet_lithology.npy")
    return np.load(path)


def make_colormap():
    """Build a discrete colormap for lithology classes."""
    codes = sorted(LITHOLOGY_MAP.keys())
    colors = [LITHOLOGY_COLORS[c] for c in codes]
    cmap = ListedColormap(colors)
    bounds = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, codes


def voxel_to_coords(i, j, k):
    """Convert voxel indices to real-world coordinates."""
    x = ORIGIN[0] + i * SPACING[0]
    y = ORIGIN[1] + j * SPACING[1]
    z = ORIGIN[2] + k * SPACING[2]
    return x, y, z


def plot_xy_slice(volume, k_index, save=True):
    """Plot a horizontal (map view) slice at depth index k."""
    cmap, norm, codes = make_colormap()
    slc = volume[:, :, k_index].T  # transpose so X=columns, Y=rows

    x_min = ORIGIN[0]
    x_max = ORIGIN[0] + (NX - 1) * SPACING[0]
    y_min = ORIGIN[1] + (NY - 1) * SPACING[1]
    y_max = ORIGIN[1]
    z_val = ORIGIN[2] + k_index * SPACING[2]

    fig, ax = plt.subplots(figsize=(14, 10))
    slc_masked = np.ma.masked_equal(slc, NODATA)
    im = ax.imshow(slc_masked, cmap=cmap, norm=norm,
                   extent=[x_min, x_max, y_min, y_max],
                   aspect='equal', origin='upper')

    cbar = plt.colorbar(im, ax=ax, ticks=codes, shrink=0.8)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Adavale Basin - Map View at Z = {z_val:.0f} m (k={k_index})")

    if save:
        out = os.path.join(OUTPUT_DIR, f"slice_xy_k{k_index}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)
    return fig


def plot_xz_cross_section(volume, j_index, save=True):
    """Plot a W-E cross-section at northing index j."""
    cmap, norm, codes = make_colormap()
    slc = volume[:, j_index, :].T  # (NZ, NX)

    x_min = ORIGIN[0]
    x_max = ORIGIN[0] + (NX - 1) * SPACING[0]
    z_min = ORIGIN[2] + (NZ - 1) * SPACING[2]
    z_max = ORIGIN[2]
    y_val = ORIGIN[1] + j_index * SPACING[1]

    fig, ax = plt.subplots(figsize=(16, 6))
    slc_masked = np.ma.masked_equal(slc, NODATA)
    im = ax.imshow(slc_masked, cmap=cmap, norm=norm,
                   extent=[x_min, x_max, z_min, z_max],
                   aspect='auto', origin='upper')

    cbar = plt.colorbar(im, ax=ax, ticks=codes, shrink=0.9)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Adavale Basin - W-E Cross Section at Y = {y_val:.0f} m (j={j_index})")

    if save:
        out = os.path.join(OUTPUT_DIR, f"section_xz_j{j_index}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)
    return fig


def plot_yz_cross_section(volume, i_index, save=True):
    """Plot a S-N cross-section at easting index i."""
    cmap, norm, codes = make_colormap()
    slc = volume[i_index, :, :].T  # (NZ, NY)

    y_min = ORIGIN[1] + (NY - 1) * SPACING[1]
    y_max = ORIGIN[1]
    z_min = ORIGIN[2] + (NZ - 1) * SPACING[2]
    z_max = ORIGIN[2]
    x_val = ORIGIN[0] + i_index * SPACING[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    slc_masked = np.ma.masked_equal(slc, NODATA)
    im = ax.imshow(slc_masked, cmap=cmap, norm=norm,
                   extent=[y_min, y_max, z_min, z_max],
                   aspect='auto', origin='upper')

    cbar = plt.colorbar(im, ax=ax, ticks=codes, shrink=0.9)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Adavale Basin - S-N Cross Section at X = {x_val:.0f} m (i={i_index})")

    if save:
        out = os.path.join(OUTPUT_DIR, f"section_yz_i{i_index}.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)
    return fig


def plot_formation_masks(save=True):
    """Plot all formation boundary polygons in map view."""
    fig, ax = plt.subplots(figsize=(14, 10))
    colors_list = list(LITHOLOGY_COLORS.values())

    for idx, (name, filename) in enumerate(MASK_FILES.items()):
        df = pd.read_csv(os.path.join(MASKS_DIR, filename))
        code = [k for k, v in LITHOLOGY_MAP.items() if v == name][0]
        color = LITHOLOGY_COLORS[code]
        # Close the polygon
        xs = list(df['X']) + [df['X'].iloc[0]]
        ys = list(df['Y']) + [df['Y'].iloc[0]]
        ax.fill(xs, ys, alpha=0.3, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.5, label=name)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Adavale Basin - Formation Distribution Masks")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')

    if save:
        out = os.path.join(OUTPUT_DIR, "formation_masks_map.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)
    return fig


def plot_class_distribution(volume, save=True):
    """Bar chart of lithology class counts."""
    valid = volume[volume != NODATA]
    unique, counts = np.unique(valid, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    names = [LITHOLOGY_MAP.get(int(u), f"?{int(u)}") for u in unique]
    colors = [LITHOLOGY_COLORS.get(int(u), (0.5, 0.5, 0.5)) for u in unique]
    bars = ax.bar(names, counts, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylabel("Voxel Count")
    ax.set_title("Lithology Class Distribution")
    ax.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count/1e6:.1f}M', ha='center', va='bottom', fontsize=8)

    if save:
        out = os.path.join(OUTPUT_DIR, "class_distribution.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)
    return fig


def visualize_3d_pyvista(volume):
    """Create 3D visualization using PyVista (saves to file)."""
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except ImportError:
        print("PyVista not installed, skipping 3D visualization.")
        return

    print("\nBuilding PyVista grid...")

    # PyVista ImageData expects (NX, NY, NZ) with positive spacing
    # We need to handle the negative Y and Z spacing
    grid = pv.ImageData()
    grid.dimensions = (NX + 1, NY + 1, NZ + 1)
    grid.origin = (ORIGIN[0], ORIGIN[1] + (NY - 1) * SPACING[1], ORIGIN[2] + (NZ - 1) * SPACING[2])
    grid.spacing = (abs(SPACING[0]), abs(SPACING[1]), abs(SPACING[2]))

    # Flip Y and Z axes in the data to match positive spacing
    vol_flipped = volume[:, ::-1, ::-1].copy()
    # PyVista expects Fortran order for cell data
    grid.cell_data["lithology"] = vol_flipped.flatten(order='F')

    # Replace nodata with NaN for clean rendering
    lith = grid.cell_data["lithology"].copy()
    lith[lith == NODATA] = np.nan
    grid.cell_data["lithology"] = lith

    # Threshold to remove NaN cells
    grid_valid = grid.threshold(value=(0.5, 13.5), scalars="lithology")

    # Build custom colormap
    codes = sorted(LITHOLOGY_COLORS.keys())
    colors_rgb = [LITHOLOGY_COLORS[c] for c in codes]
    cmap = ListedColormap(colors_rgb)

    # Render orthogonal slices
    print("Rendering 3D views...")

    # Full 3D volume render
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.add_mesh(grid_valid, scalars="lithology", cmap=cmap,
                     clim=[0.5, 13.5], show_edges=False,
                     scalar_bar_args={"title": "Lithology"})
    plotter.add_axes()
    plotter.camera_position = 'iso'
    out = os.path.join(OUTPUT_DIR, "3d_volume_iso.png")
    plotter.screenshot(out)
    print(f"Saved: {out}")
    plotter.close()

    # Top-down view
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.add_mesh(grid_valid, scalars="lithology", cmap=cmap,
                     clim=[0.5, 13.5], show_edges=False)
    plotter.view_xy()
    plotter.add_axes()
    out = os.path.join(OUTPUT_DIR, "3d_volume_top.png")
    plotter.screenshot(out)
    print(f"Saved: {out}")
    plotter.close()

    # Save the grid as VTI for later use
    vti_path = os.path.join(OUTPUT_DIR, "adavale_lithology.vti")
    grid.save(vti_path)
    print(f"Saved VTI: {vti_path}")

    return grid


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volume = load_volume()
    print(f"Volume loaded: {volume.shape}")

    # 2D slices at key depths
    print("\n--- Horizontal slices (map view) ---")
    # k=0 is top (z=690m), k=167 is ~mid depth, k=334 is bottom
    for k in [0, 50, 100, 167, 250, 334]:
        z_val = ORIGIN[2] + k * SPACING[2]
        print(f"  k={k} -> z={z_val:.0f}m")
        plot_xy_slice(volume, k)

    print("\n--- W-E cross sections ---")
    for j in [150, 317, 500]:
        y_val = ORIGIN[1] + j * SPACING[1]
        print(f"  j={j} -> y={y_val:.0f}m")
        plot_xz_cross_section(volume, j)

    print("\n--- S-N cross sections ---")
    for i in [128, 256, 384]:
        x_val = ORIGIN[0] + i * SPACING[0]
        print(f"  i={i} -> x={x_val:.0f}m")
        plot_yz_cross_section(volume, i)

    print("\n--- Formation masks ---")
    plot_formation_masks()

    print("\n--- Class distribution ---")
    plot_class_distribution(volume)

    print("\n--- 3D PyVista visualization ---")
    visualize_3d_pyvista(volume)

    print("\nDone! All outputs in:", OUTPUT_DIR)
