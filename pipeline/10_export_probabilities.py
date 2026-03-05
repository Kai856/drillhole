"""Export predictions with softmax probabilities for web viewer.

Re-runs inference on the subsampled grid, saving top-class + full probability
distribution so the web UI can show confidence per voxel and area-level stats.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import NX, NY, NZ, ORIGIN, SPACING, LITHOLOGY_MAP, OUTPUT_DIR

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(OUTPUT_DIR, "realistic_model")

# Same model architecture
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

class NeuralField(nn.Module):
    def __init__(self, n_classes=13, n_freqs=10, hidden_dim=256, n_layers=8):
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


def main():
    print(f"Device: {DEVICE}")

    # Load model
    model = NeuralField(n_classes=13, n_freqs=10, hidden_dim=256, n_layers=8)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "best_model.pt"),
        map_location=DEVICE, weights_only=True
    ))
    model = model.to(DEVICE)
    model.eval()

    # Load normalization
    with open(os.path.join(MODEL_DIR, "norm_params.json")) as f:
        norm = json.load(f)
    coord_min = np.array(norm["coord_min"])
    coord_max = np.array(norm["coord_max"])
    rng = coord_max - coord_min

    # Scene transform (must match index.html)
    SC_xc, SC_yc, SC_zc = 297750, 7172250, -2710
    SC_scale = 0.0006349206349206349
    SC_zScale = 0.0125

    step = 10  # subsample
    cubeW = abs(SPACING[0]) * step * SC_scale
    cubeH = abs(SPACING[2]) * step * SC_zScale
    cubeD = abs(SPACING[1]) * step * SC_scale

    print(f"Grid: {NX}x{NY}x{NZ}, step={step}")
    print(f"Cube size: [{cubeW:.4f}, {cubeH:.4f}, {cubeD:.4f}]")

    # Build all coords at subsampled resolution
    ii = np.arange(0, NX, step)
    jj = np.arange(0, NY, step)
    kk = np.arange(0, NZ, step)
    print(f"Subsampled: {len(ii)}x{len(jj)}x{len(kk)} = {len(ii)*len(jj)*len(kk):,} voxels")

    # Per-formation storage: x,y,z in scene coords + avg probability
    formations = {str(c): {"x": [], "y": [], "z": [], "prob": []} for c in range(1, 14)}

    with torch.no_grad():
        for idx_i, i in enumerate(ii):
            # Build coords for this slice
            jjg, kkg = np.meshgrid(jj, kk, indexing='ij')
            coords = np.empty((len(jj) * len(kk), 3), dtype=np.float32)
            coords[:, 0] = ORIGIN[0] + i * SPACING[0]
            coords[:, 1] = ORIGIN[1] + jjg.ravel() * SPACING[1]
            coords[:, 2] = ORIGIN[2] + kkg.ravel() * SPACING[2]

            # Normalize
            coords_norm = np.empty_like(coords)
            coords_norm[:, 0] = 2.0 * (coords[:, 0] - coord_min[0]) / rng[0] - 1.0
            coords_norm[:, 1] = 2.0 * (coords[:, 1] - coord_min[1]) / rng[1] - 1.0
            coords_norm[:, 2] = 2.0 * (coords[:, 2] - coord_min[2]) / rng[2] - 1.0

            ct = torch.from_numpy(coords_norm).to(DEVICE)

            # Get logits and softmax
            logits = model(ct)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)  # 0-indexed
            max_probs = probs.max(axis=-1)

            # Convert to scene coords
            wx = coords[:, 0]
            wy = coords[:, 1]
            wz = coords[:, 2]
            sx = (wx - SC_xc) * SC_scale
            sy = (wz - SC_zc) * SC_zScale
            sz = (wy - SC_yc) * SC_scale

            for n in range(len(preds)):
                code = int(preds[n]) + 1
                formations[str(code)]["x"].append(round(float(sx[n]), 4))
                formations[str(code)]["y"].append(round(float(sz[n]), 4))
                formations[str(code)]["z"].append(round(float(sy[n]), 4))
                formations[str(code)]["prob"].append(round(float(max_probs[n]), 3))

            if (idx_i + 1) % 10 == 0:
                print(f"  Slice {idx_i+1}/{len(ii)}")

    total = sum(len(f["x"]) for f in formations.values())
    print(f"\nTotal voxels: {total:,}")

    # Print formation distribution
    for code in range(1, 14):
        n = len(formations[str(code)]["x"])
        if n > 0:
            avg_prob = np.mean(formations[str(code)]["prob"])
            print(f"  {LITHOLOGY_MAP[code]:<16} {n:>8,} voxels  avg conf: {avg_prob:.3f}")

    output = {
        "scene": {"cubeSize": [cubeW, cubeD, cubeH]},
        "formations": formations,
    }

    out_path = os.path.join(OUTPUT_DIR, "web", "realistic_predictions.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
