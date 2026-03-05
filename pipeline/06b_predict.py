"""Re-run inference with the trained neural field model."""
import numpy as np
import torch
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import NX, NY, NZ, ORIGIN, SPACING, NODATA, OUTPUT_DIR

# Import model class
from neural_field_classes import NeuralField

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_DIR = os.path.join(OUTPUT_DIR, "neural_field")


def predict_volume(model):
    model = model.to(DEVICE)
    model.eval()

    with open(os.path.join(MODEL_DIR, "norm_params.json")) as f:
        norm = json.load(f)
    coord_min = np.array(norm["coord_min"])
    coord_max = np.array(norm["coord_max"])
    rng = coord_max - coord_min

    print(f"Predicting ({NX}x{NY}x{NZ}) = {NX*NY*NZ:,} points on {DEVICE}...")
    predicted = np.zeros((NX, NY, NZ), dtype=np.int8)

    with torch.no_grad():
        for i in range(NX):
            jj, kk = np.meshgrid(np.arange(NY), np.arange(NZ), indexing='ij')
            coords = np.empty((NY * NZ, 3), dtype=np.float32)
            coords[:, 0] = ORIGIN[0] + i * SPACING[0]
            coords[:, 1] = ORIGIN[1] + jj.ravel() * SPACING[1]
            coords[:, 2] = ORIGIN[2] + kk.ravel() * SPACING[2]

            coords_norm = np.empty_like(coords)
            coords_norm[:, 0] = 2.0 * (coords[:, 0] - coord_min[0]) / rng[0] - 1.0
            coords_norm[:, 1] = 2.0 * (coords[:, 1] - coord_min[1]) / rng[1] - 1.0
            coords_norm[:, 2] = 2.0 * (coords[:, 2] - coord_min[2]) / rng[2] - 1.0

            ct = torch.from_numpy(coords_norm).to(DEVICE)

            preds = []
            for j in range(0, len(ct), 200000):
                preds.append(model(ct[j:j+200000]).argmax(dim=-1).cpu().numpy())
            predicted[i] = np.concatenate(preds).reshape(NY, NZ) + 1

            if (i + 1) % 100 == 0:
                print(f"  Slice {i+1}/{NX}")

    out = os.path.join(MODEL_DIR, "predicted_volume.npy")
    np.save(out, predicted)
    print(f"Saved: {out}")
    return predicted


if __name__ == "__main__":
    model = NeuralField(n_classes=13, n_freqs=10, hidden_dim=256, n_layers=5)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                                      map_location=DEVICE, weights_only=True))
    predicted = predict_volume(model)

    # Verify
    gt = np.load(os.path.join(OUTPUT_DIR, "voxet_lithology.npy"))
    valid = gt != 0
    acc = (gt[valid] == predicted[valid]).mean()
    print(f"\nOverall accuracy (all voxels): {acc:.4f}")
