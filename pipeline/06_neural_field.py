"""Neural Field: learns f(x, y, z) → lithology class for the Adavale Basin.

Optimized for speed: coordinates computed on-the-fly from voxel indices,
training subsampled per epoch, large batches.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, OUTPUT_DIR
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(OUTPUT_DIR, "neural_field")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, n_freqs=10, input_dim=3):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)
        self.output_dim = input_dim + input_dim * 2 * n_freqs

    def forward(self, x):
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)


# ─── Model ──────────────────────────────────────────────────────────────────

class NeuralField(nn.Module):
    def __init__(self, n_classes=13, n_freqs=10, hidden_dim=256, n_layers=5):
        super().__init__()
        self.encoder = PositionalEncoding(n_freqs=n_freqs)

        mid = n_layers // 2
        self.pre_skip = nn.ModuleList()
        self.post_skip = nn.ModuleList()

        enc_dim = self.encoder.output_dim
        for i in range(n_layers):
            if i < mid:
                self.pre_skip.append(nn.Linear(enc_dim if i == 0 else hidden_dim, hidden_dim))
            elif i == mid:
                self.post_skip.append(nn.Linear(hidden_dim + enc_dim, hidden_dim))
            else:
                self.post_skip.append(nn.Linear(hidden_dim, hidden_dim))

        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, coords):
        enc = self.encoder(coords)
        h = enc
        for layer in self.pre_skip:
            h = torch.relu(layer(h))
        h = torch.cat([h, enc], dim=-1)
        for layer in self.post_skip:
            h = torch.relu(layer(h))
        return self.head(h)


# ─── Fast data pipeline ─────────────────────────────────────────────────────

def prepare_data(volume):
    """Build flat arrays of valid voxel indices and labels. Lightweight."""
    print("Preparing data...")
    valid_mask = volume != NODATA
    # Flat indices of valid voxels
    flat_indices = np.flatnonzero(valid_mask)
    labels = volume.flat[flat_indices].astype(np.int64) - 1  # 0-12

    # Precompute normalization bounds (real-world coords of grid corners)
    coord_min = np.array([
        ORIGIN[0],
        ORIGIN[1] + (NY - 1) * SPACING[1],  # Y decreases
        ORIGIN[2] + (NZ - 1) * SPACING[2],  # Z decreases
    ])
    coord_max = np.array([
        ORIGIN[0] + (NX - 1) * SPACING[0],
        ORIGIN[1],
        ORIGIN[2],
    ])

    norm_params = {"coord_min": coord_min.tolist(), "coord_max": coord_max.tolist()}
    with open(os.path.join(MODEL_DIR, "norm_params.json"), "w") as f:
        json.dump(norm_params, f)

    # ── Drill-hole holdout ──
    # Scatter N "wells" across the basin. Each well is a 10x10 cell block
    # (5km x 5km) through all depths. Hold these out as test set.
    n_wells = 100
    well_radius = 5  # half-width in grid cells (5 cells = 2.5km each side)

    rng = np.random.RandomState(42)
    # Place well centers avoiding edges (need radius margin)
    margin = well_radius
    well_centers_i = rng.randint(margin, NX - margin, size=n_wells)
    well_centers_j = rng.randint(margin, NY - margin, size=n_wells)

    # Build a 2D mask of held-out XY cells
    well_mask_2d = np.zeros((NX, NY), dtype=bool)
    for ci, cj in zip(well_centers_i, well_centers_j):
        well_mask_2d[ci-well_radius:ci+well_radius,
                     cj-well_radius:cj+well_radius] = True

    # Expand to 3D: each held-out XY column covers all Z
    ijk = np.column_stack(np.unravel_index(flat_indices, (NX, NY, NZ)))
    val_block = well_mask_2d[ijk[:, 0], ijk[:, 1]]

    val_idx = np.where(val_block)[0]
    train_idx = np.where(~val_block)[0]

    # Compute well locations in real-world coords for reporting
    well_x = ORIGIN[0] + well_centers_i * SPACING[0]
    well_y = ORIGIN[1] + well_centers_j * SPACING[1]
    xy_cells_held = well_mask_2d.sum()

    print(f"  Drill-hole holdout: {n_wells} wells, {well_radius*2}x{well_radius*2} cells each (5km x 5km)")
    print(f"  Unique XY cells held out: {xy_cells_held:,} / {NX*NY:,} ({xy_cells_held/(NX*NY)*100:.1f}%)")

    # Class weights (from training set only)
    train_labels = labels[train_idx]
    unique, counts = np.unique(train_labels, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(unique)

    val_labels = labels[val_idx]
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    print(f"\n  Valid voxels: {len(labels):,} | Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    print(f"  Val fraction: {len(val_idx)/len(labels)*100:.1f}%")
    print(f"  Val classes present: {[LITHOLOGY_MAP[u+1] for u in val_unique]}")
    print(f"  Class weights: { {LITHOLOGY_MAP[u+1]: round(w,2) for u,w in zip(unique,weights)} }")

    # Save split info + well locations
    split_info = {
        "method": "drill_hole_holdout",
        "n_wells": n_wells,
        "well_radius_cells": well_radius,
        "well_block_size_km": well_radius * abs(SPACING[0]) / 1000 * 2,
        "well_centers_i": well_centers_i.tolist(),
        "well_centers_j": well_centers_j.tolist(),
        "well_x": well_x.tolist(),
        "well_y": well_y.tolist(),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    with open(os.path.join(MODEL_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    return flat_indices, labels, train_idx, val_idx, coord_min, coord_max, \
           torch.tensor(weights, dtype=torch.float32)


def indices_to_coords_norm(flat_indices, coord_min, coord_max):
    """Convert flat voxel indices to normalized [-1,1] coordinates. Fast vectorized."""
    # Unravel flat index to (i, j, k) — volume is (NX, NY, NZ) Fortran order
    # But numpy default is C order, so unravel accordingly
    ijk = np.column_stack(np.unravel_index(flat_indices, (NX, NY, NZ)))

    # Real-world coords
    coords = np.empty_like(ijk, dtype=np.float32)
    coords[:, 0] = ORIGIN[0] + ijk[:, 0] * SPACING[0]
    coords[:, 1] = ORIGIN[1] + ijk[:, 1] * SPACING[1]
    coords[:, 2] = ORIGIN[2] + ijk[:, 2] * SPACING[2]

    # Normalize to [-1, 1]
    rng = coord_max - coord_min
    coords[:, 0] = 2.0 * (coords[:, 0] - coord_min[0]) / rng[0] - 1.0
    coords[:, 1] = 2.0 * (coords[:, 1] - coord_min[1]) / rng[1] - 1.0
    coords[:, 2] = 2.0 * (coords[:, 2] - coord_min[2]) / rng[2] - 1.0

    return coords


def make_batches(flat_indices, labels, idx_subset, coord_min, coord_max,
                 batch_size, shuffle=True, max_samples=None):
    """Generator that yields (coords_tensor, labels_tensor) batches.

    Computes coordinates on-the-fly from flat indices. If max_samples is set,
    randomly subsample the subset each call (for faster epochs).
    """
    if max_samples and max_samples < len(idx_subset):
        chosen = np.random.choice(idx_subset, max_samples, replace=False)
    else:
        chosen = idx_subset

    if shuffle:
        chosen = np.random.permutation(chosen)

    for start in range(0, len(chosen), batch_size):
        batch_idx = chosen[start:start + batch_size]
        fi = flat_indices[batch_idx]
        lab = labels[batch_idx]
        coords = indices_to_coords_norm(fi, coord_min, coord_max)
        yield (torch.from_numpy(coords).to(DEVICE),
               torch.from_numpy(lab).to(DEVICE))


# ─── Training ───────────────────────────────────────────────────────────────

def train_model(model, flat_indices, labels, train_idx, val_idx,
                coord_min, coord_max, class_weights,
                epochs=30, batch_size=131072, lr=1e-3,
                train_samples_per_epoch=10_000_000,
                val_samples=2_000_000):
    """Train with subsampled epochs for speed."""
    model = model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "per_class_acc": []}

    print(f"\nTraining on {DEVICE}")
    print(f"Batch: {batch_size:,} | Train/epoch: {train_samples_per_epoch/1e6:.0f}M | Val: {val_samples/1e6:.0f}M")
    print("-" * 70)

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        epoch_loss = 0
        n_batches = 0
        for coords, lab in make_batches(flat_indices, labels, train_idx,
                                         coord_min, coord_max, batch_size,
                                         shuffle=True, max_samples=train_samples_per_epoch):
            logits = model(coords)
            loss = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        class_correct = np.zeros(13)
        class_total = np.zeros(13)
        n_vb = 0

        with torch.no_grad():
            for coords, lab in make_batches(flat_indices, labels, val_idx,
                                             coord_min, coord_max, batch_size * 2,
                                             shuffle=False, max_samples=val_samples):
                logits = model(coords)
                loss = criterion(logits, lab)
                val_loss += loss.item()
                n_vb += 1
                preds = logits.argmax(dim=-1)
                correct += (preds == lab).sum().item()
                total += lab.size(0)
                for c in range(13):
                    m = lab == c
                    class_total[c] += m.sum().item()
                    class_correct[c] += (preds[m] == c).sum().item()

        val_loss /= max(n_vb, 1)
        val_acc = correct / max(total, 1)

        per_class = {}
        for c in range(13):
            if class_total[c] > 0:
                per_class[LITHOLOGY_MAP[c + 1]] = round(class_correct[c] / class_total[c], 4)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["per_class_acc"].append(per_class)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:>3}/{epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | Best: {best_val_acc:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 5 == 0:
            for name, acc in per_class.items():
                bar = "█" * int(acc * 30)
                print(f"    {name:<10} {acc:.3f} {bar}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pt"))
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history


# ─── Inference ──────────────────────────────────────────────────────────────

def predict_volume(model, resolution_factor=1):
    """Predict lithology over the full grid using exact same coord mapping as training."""
    model = model.to(DEVICE)
    model.eval()

    with open(os.path.join(MODEL_DIR, "norm_params.json")) as f:
        norm = json.load(f)
    coord_min = np.array(norm["coord_min"])
    coord_max = np.array(norm["coord_max"])
    rng = coord_max - coord_min

    nx, ny, nz = NX, NY, NZ
    print(f"Predicting ({nx}x{ny}x{nz}) = {nx*ny*nz:,} points...")
    predicted = np.zeros((nx, ny, nz), dtype=np.int8)

    with torch.no_grad():
        for i in range(nx):
            # Build (j, k) grid for this i-slice
            jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
            # Compute real-world coords exactly like training does
            coords = np.empty((ny * nz, 3), dtype=np.float32)
            coords[:, 0] = ORIGIN[0] + i * SPACING[0]
            coords[:, 1] = ORIGIN[1] + jj.ravel() * SPACING[1]
            coords[:, 2] = ORIGIN[2] + kk.ravel() * SPACING[2]

            # Normalize exactly like training
            coords_norm = np.empty_like(coords)
            coords_norm[:, 0] = 2.0 * (coords[:, 0] - coord_min[0]) / rng[0] - 1.0
            coords_norm[:, 1] = 2.0 * (coords[:, 1] - coord_min[1]) / rng[1] - 1.0
            coords_norm[:, 2] = 2.0 * (coords[:, 2] - coord_min[2]) / rng[2] - 1.0

            ct = torch.from_numpy(coords_norm).to(DEVICE)

            preds = []
            for j in range(0, len(ct), 200000):
                preds.append(model(ct[j:j+200000]).argmax(dim=-1).cpu().numpy())
            predicted[i] = np.concatenate(preds).reshape(ny, nz) + 1  # back to 1-13

            if (i + 1) % 100 == 0:
                print(f"  Slice {i+1}/{nx}")

    out = os.path.join(MODEL_DIR, "predicted_volume.npy")
    np.save(out, predicted)
    print(f"Saved: {out}")
    return predicted


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    volume = np.load(os.path.join(OUTPUT_DIR, "voxet_lithology.npy"))
    print(f"Volume: {volume.shape}\n")

    flat_indices, labels, train_idx, val_idx, coord_min, coord_max, class_weights = \
        prepare_data(volume)

    del volume  # free ~400MB

    model = NeuralField(n_classes=13, n_freqs=10, hidden_dim=256, n_layers=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters\n")

    history = train_model(
        model, flat_indices, labels, train_idx, val_idx,
        coord_min, coord_max, class_weights,
        epochs=30,
        batch_size=131072,
        lr=1e-3,
        train_samples_per_epoch=10_000_000,  # 10M per epoch (not all 82M)
        val_samples=2_000_000,               # 2M for validation
    )

    print("\n--- Full volume inference ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    predict_volume(model)
    print("\nDone!")
