"""Compare ground truth vs neural field predictions at held-out drill holes."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, OUTPUT_DIR
)

MODEL_DIR = os.path.join(OUTPUT_DIR, "neural_field")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "comparison")
os.makedirs(COMPARE_DIR, exist_ok=True)


def make_colormap():
    codes = sorted(LITHOLOGY_MAP.keys())
    colors = [LITHOLOGY_COLORS[c] for c in codes]
    cmap = ListedColormap(colors)
    bounds = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, codes


def load_data():
    gt = np.load(os.path.join(OUTPUT_DIR, "voxet_lithology.npy"))
    pred = np.load(os.path.join(MODEL_DIR, "predicted_volume.npy"))
    with open(os.path.join(MODEL_DIR, "split_info.json")) as f:
        split = json.load(f)
    return gt, pred, split


def get_well_mask_2d(split):
    """Rebuild the 2D boolean mask of held-out well locations."""
    mask = np.zeros((NX, NY), dtype=bool)
    radius = split["well_radius_cells"]
    for ci, cj in zip(split["well_centers_i"], split["well_centers_j"]):
        mask[ci-radius:ci+radius, cj-radius:cj+radius] = True
    return mask


# ─── Horizontal slice (map view) ────────────────────────────────────────────

def plot_xy_comparison(gt, pred, well_mask_2d, k_index, split):
    """Side-by-side map view: ground truth | prediction | error at drill holes."""
    cmap, norm, codes = make_colormap()

    z_val = ORIGIN[2] + k_index * SPACING[2]
    x_min, x_max = ORIGIN[0], ORIGIN[0] + (NX - 1) * SPACING[0]
    y_min = ORIGIN[1] + (NY - 1) * SPACING[1]
    y_max = ORIGIN[1]
    extent = [x_min, x_max, y_min, y_max]

    gt_slice = gt[:, :, k_index].T.astype(float)
    pred_slice = pred[:, :, k_index].T.astype(float)
    gt_slice[gt_slice == NODATA] = np.nan
    pred_slice[pred_slice == NODATA] = np.nan

    # Error map: 1 where wrong, 0 where correct, NaN where nodata
    well_mask_t = well_mask_2d.T  # match slice orientation
    error = np.full_like(gt_slice, np.nan)
    valid = ~np.isnan(gt_slice) & well_mask_t
    error[valid] = (gt_slice[valid] != pred_slice[valid]).astype(float)

    # Prediction only at wells (rest is NaN)
    pred_at_wells = np.full_like(pred_slice, np.nan)
    pred_at_wells[well_mask_t] = pred_slice[well_mask_t]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Ground truth with well outlines
    ax = axes[0]
    gt_masked = np.ma.masked_invalid(gt_slice)
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='equal', origin='upper')
    # Draw well block outlines
    radius = split["well_radius_cells"]
    for ci, cj in zip(split["well_centers_i"], split["well_centers_j"]):
        wx = ORIGIN[0] + (ci - radius) * SPACING[0]
        wy = ORIGIN[1] + (cj - radius) * SPACING[1]
        w = radius * 2 * abs(SPACING[0])
        h = radius * 2 * abs(SPACING[1])
        rect = Rectangle((wx, wy - h), w, h, linewidth=0.5,
                          edgecolor='white', facecolor='none', alpha=0.6)
        ax.add_patch(rect)
    ax.set_title("Ground Truth + Well Locations", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Prediction at well locations (overlay on faded ground truth)
    ax = axes[1]
    # Faded ground truth as background
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='equal',
              origin='upper', alpha=0.2)
    # Bright prediction at wells
    pred_masked = np.ma.masked_invalid(pred_at_wells)
    ax.imshow(pred_masked, cmap=cmap, norm=norm, extent=extent, aspect='equal', origin='upper')
    ax.set_title("Model Prediction at Wells", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Error map at wells
    ax = axes[2]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='equal',
              origin='upper', alpha=0.15)
    error_cmap = ListedColormap(['#2ecc71', '#e74c3c'])  # green=correct, red=wrong
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    error_masked = np.ma.masked_invalid(error)
    ax.imshow(error_masked, cmap=error_cmap, norm=error_norm, extent=extent,
              aspect='equal', origin='upper')
    n_correct = np.nansum(error == 0)
    n_wrong = np.nansum(error == 1)
    total = n_correct + n_wrong
    acc = n_correct / total * 100 if total > 0 else 0
    ax.set_title(f"Errors at Wells ({acc:.1f}% correct)", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes[:2], ticks=codes, shrink=0.8, pad=0.02)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)

    fig.suptitle(f"Adavale Basin — Map View at Z = {z_val:.0f} m (k={k_index})", fontsize=14, y=1.02)
    fig.tight_layout()

    out = os.path.join(COMPARE_DIR, f"compare_xy_k{k_index}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ─── W-E Cross Section ──────────────────────────────────────────────────────

def plot_xz_comparison(gt, pred, well_mask_2d, j_index, split):
    """W-E cross section: ground truth | prediction at wells | error."""
    cmap, norm, codes = make_colormap()

    y_val = ORIGIN[1] + j_index * SPACING[1]
    x_min, x_max = ORIGIN[0], ORIGIN[0] + (NX - 1) * SPACING[0]
    z_min = ORIGIN[2] + (NZ - 1) * SPACING[2]
    z_max = ORIGIN[2]
    extent = [x_min, x_max, z_min, z_max]

    gt_slice = gt[:, j_index, :].T.astype(float)  # (NZ, NX)
    pred_slice = pred[:, j_index, :].T.astype(float)
    gt_slice[gt_slice == NODATA] = np.nan
    pred_slice[pred_slice == NODATA] = np.nan

    # Which X columns are held-out wells at this J
    well_cols = well_mask_2d[:, j_index]  # (NX,) bool

    # Prediction only at well columns
    pred_at_wells = np.full_like(pred_slice, np.nan)
    pred_at_wells[:, well_cols] = pred_slice[:, well_cols]

    # Error at wells
    error = np.full_like(gt_slice, np.nan)
    valid = ~np.isnan(gt_slice)
    for i in range(NX):
        if well_cols[i]:
            col_valid = valid[:, i]
            error[col_valid, i] = (gt_slice[col_valid, i] != pred_slice[col_valid, i]).astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    # Ground truth with well column markers
    ax = axes[0]
    gt_masked = np.ma.masked_invalid(gt_slice)
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto', origin='upper')
    # Mark well columns
    radius = split["well_radius_cells"]
    for ci, cj in zip(split["well_centers_i"], split["well_centers_j"]):
        if abs(cj - j_index) < radius:
            wx = ORIGIN[0] + (ci - radius) * SPACING[0]
            w = radius * 2 * abs(SPACING[0])
            ax.axvspan(wx, wx + w, alpha=0.15, color='white')
    ax.set_title("Ground Truth + Well Locations", fontsize=12)
    ax.set_ylabel("Elevation (m)")

    # Prediction at wells
    ax = axes[1]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto',
              origin='upper', alpha=0.2)
    pred_masked = np.ma.masked_invalid(pred_at_wells)
    ax.imshow(pred_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto', origin='upper')
    ax.set_title("Model Prediction at Wells", fontsize=12)
    ax.set_ylabel("Elevation (m)")

    # Error
    ax = axes[2]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto',
              origin='upper', alpha=0.15)
    error_cmap = ListedColormap(['#2ecc71', '#e74c3c'])
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    error_masked = np.ma.masked_invalid(error)
    ax.imshow(error_masked, cmap=error_cmap, norm=error_norm, extent=extent,
              aspect='auto', origin='upper')
    n_correct = np.nansum(error == 0)
    n_wrong = np.nansum(error == 1)
    total = n_correct + n_wrong
    acc = n_correct / total * 100 if total > 0 else 0
    ax.set_title(f"Errors at Wells (green=correct, red=wrong, {acc:.1f}% acc)", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Elevation (m)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, ticks=codes, shrink=0.6, pad=0.02)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)

    fig.suptitle(f"W-E Cross Section at Y = {y_val:.0f} m (j={j_index})", fontsize=14)
    fig.tight_layout()

    out = os.path.join(COMPARE_DIR, f"compare_xz_j{j_index}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ─── S-N Cross Section ──────────────────────────────────────────────────────

def plot_yz_comparison(gt, pred, well_mask_2d, i_index, split):
    """S-N cross section: ground truth | prediction at wells | error."""
    cmap, norm, codes = make_colormap()

    x_val = ORIGIN[0] + i_index * SPACING[0]
    y_min = ORIGIN[1] + (NY - 1) * SPACING[1]
    y_max = ORIGIN[1]
    z_min = ORIGIN[2] + (NZ - 1) * SPACING[2]
    z_max = ORIGIN[2]
    extent = [y_min, y_max, z_min, z_max]

    gt_slice = gt[i_index, :, :].T.astype(float)  # (NZ, NY)
    pred_slice = pred[i_index, :, :].T.astype(float)
    gt_slice[gt_slice == NODATA] = np.nan
    pred_slice[pred_slice == NODATA] = np.nan

    well_cols = well_mask_2d[i_index, :]  # (NY,) bool

    pred_at_wells = np.full_like(pred_slice, np.nan)
    pred_at_wells[:, well_cols] = pred_slice[:, well_cols]

    error = np.full_like(gt_slice, np.nan)
    valid = ~np.isnan(gt_slice)
    for j in range(NY):
        if well_cols[j]:
            col_valid = valid[:, j]
            error[col_valid, j] = (gt_slice[col_valid, j] != pred_slice[col_valid, j]).astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    gt_masked = np.ma.masked_invalid(gt_slice)

    ax = axes[0]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto', origin='upper')
    radius = split["well_radius_cells"]
    for ci, cj in zip(split["well_centers_i"], split["well_centers_j"]):
        if abs(ci - i_index) < radius:
            wy = ORIGIN[1] + (cj - radius) * SPACING[1]
            h = radius * 2 * abs(SPACING[1])
            ax.axvspan(wy - h, wy, alpha=0.15, color='white')
    ax.set_title("Ground Truth + Well Locations", fontsize=12)
    ax.set_ylabel("Elevation (m)")

    ax = axes[1]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto',
              origin='upper', alpha=0.2)
    pred_masked = np.ma.masked_invalid(pred_at_wells)
    ax.imshow(pred_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto', origin='upper')
    ax.set_title("Model Prediction at Wells", fontsize=12)
    ax.set_ylabel("Elevation (m)")

    ax = axes[2]
    ax.imshow(gt_masked, cmap=cmap, norm=norm, extent=extent, aspect='auto',
              origin='upper', alpha=0.15)
    error_cmap = ListedColormap(['#2ecc71', '#e74c3c'])
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    error_masked = np.ma.masked_invalid(error)
    ax.imshow(error_masked, cmap=error_cmap, norm=error_norm, extent=extent,
              aspect='auto', origin='upper')
    n_correct = np.nansum(error == 0)
    n_wrong = np.nansum(error == 1)
    total = n_correct + n_wrong
    acc = n_correct / total * 100 if total > 0 else 0
    ax.set_title(f"Errors at Wells ({acc:.1f}% acc)", fontsize=12)
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("Elevation (m)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, ticks=codes, shrink=0.6, pad=0.02)
    cbar.ax.set_yticklabels([LITHOLOGY_MAP[c] for c in codes], fontsize=8)

    fig.suptitle(f"S-N Cross Section at X = {x_val:.0f} m (i={i_index})", fontsize=14)
    fig.tight_layout()

    out = os.path.join(COMPARE_DIR, f"compare_yz_i{i_index}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ─── Well location overview map ─────────────────────────────────────────────

def plot_well_map(split):
    """Map showing where all 100 drill holes are located."""
    fig, ax = plt.subplots(figsize=(14, 10))

    radius = split["well_radius_cells"]
    for ci, cj, wx, wy in zip(split["well_centers_i"], split["well_centers_j"],
                                split["well_x"], split["well_y"]):
        bx = ORIGIN[0] + (ci - radius) * SPACING[0]
        by = ORIGIN[1] + (cj + radius) * SPACING[1]
        w = radius * 2 * abs(SPACING[0])
        h = radius * 2 * abs(SPACING[1])
        rect = Rectangle((bx, by), w, h, linewidth=0.5,
                          edgecolor='#e94560', facecolor='#e94560', alpha=0.4)
        ax.add_patch(rect)
        ax.plot(wx, wy, 'k.', markersize=2)

    ax.set_xlim(ORIGIN[0], ORIGIN[0] + (NX - 1) * SPACING[0])
    ax.set_ylim(ORIGIN[1] + (NY - 1) * SPACING[1], ORIGIN[1])
    ax.set_aspect('equal')
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Held-Out Well Locations ({split['n_wells']} wells, "
                 f"{split['well_block_size_km']:.0f}km x {split['well_block_size_km']:.0f}km each)")
    ax.grid(True, alpha=0.3)

    out = os.path.join(COMPARE_DIR, "well_locations.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ─── Per-class accuracy bar chart ───────────────────────────────────────────

def plot_accuracy_comparison(gt, pred, well_mask_2d):
    """Per-class accuracy at held-out wells."""
    # Flatten to just well voxels
    well_mask_3d = np.repeat(well_mask_2d[:, :, np.newaxis], NZ, axis=2)
    valid = (gt != NODATA) & well_mask_3d

    gt_vals = gt[valid].astype(int)
    pred_vals = pred[valid].astype(int)

    codes = sorted(LITHOLOGY_MAP.keys())
    names = [LITHOLOGY_MAP[c] for c in codes]
    colors = [LITHOLOGY_COLORS[c] for c in codes]
    accs = []
    counts = []
    for c in codes:
        mask = gt_vals == c
        n = mask.sum()
        counts.append(n)
        if n > 0:
            accs.append((pred_vals[mask] == c).sum() / n)
        else:
            accs.append(0)

    overall = (gt_vals == pred_vals).sum() / len(gt_vals)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(overall, color='red', linestyle='--', linewidth=1.5, label=f'Overall: {overall:.1%}')

    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}\n({count/1e3:.0f}k)', ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy at Held-Out Wells")
    ax.set_title("Per-Formation Prediction Accuracy (Drill-Hole Holdout)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=10)

    out = os.path.join(COMPARE_DIR, "accuracy_per_class.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gt, pred, split = load_data()
    well_mask_2d = get_well_mask_2d(split)
    print(f"Ground truth: {gt.shape} | Predicted: {pred.shape}")
    print(f"Wells: {split['n_wells']} | Val voxels: {split['n_val']:,}\n")

    print("--- Well location map ---")
    plot_well_map(split)

    print("\n--- Horizontal slices ---")
    for k in [0, 50, 100, 167, 250]:
        z_val = ORIGIN[2] + k * SPACING[2]
        print(f"  k={k} z={z_val:.0f}m")
        plot_xy_comparison(gt, pred, well_mask_2d, k, split)

    print("\n--- W-E cross sections ---")
    for j in [150, 317, 500]:
        y_val = ORIGIN[1] + j * SPACING[1]
        print(f"  j={j} y={y_val:.0f}m")
        plot_xz_comparison(gt, pred, well_mask_2d, j, split)

    print("\n--- S-N cross sections ---")
    for i in [128, 256, 384]:
        x_val = ORIGIN[0] + i * SPACING[0]
        print(f"  i={i} x={x_val:.0f}m")
        plot_yz_comparison(gt, pred, well_mask_2d, i, split)

    print("\n--- Accuracy chart ---")
    plot_accuracy_comparison(gt, pred, well_mask_2d)

    print(f"\nAll comparisons saved to: {COMPARE_DIR}")
