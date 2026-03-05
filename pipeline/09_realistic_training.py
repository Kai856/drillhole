"""Realistic neural field: train ONLY on real drill hole data.

Simulates the real-world scenario: a mining company uploads drill hole logs,
and the model predicts subsurface geology everywhere else.

- 86 real drill holes from GeoModeller XML
- Train on 66 wells (~75%), hold out 20 for testing
- Compare predictions against held-out wells AND full voxel model
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
    LITHOLOGY_MAP, OUTPUT_DIR
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(OUTPUT_DIR, "realistic_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Formation name mapping: XML names -> voxel grid codes
FM_TO_CODE = {
    "GRANI": 1, "GUMBA": 2, "EASTW": 3, "LOG": 4, "BURY": 5,
    "LISSO": 6, "COOLA": 7, "BOREE": 8, "ETONV": 9, "BUCKA": 10,
    "GALILEE": 11, "EROMANGA": 12, "GLEND": 13,
    # Map extra formations to closest match
    "BASEM": 1,    # Basement -> Granite
    "SEDIM": 11,   # Sedimentary -> Galilee (generic basin sediments)
    "UNDIF": 11,   # Undifferentiated -> Galilee
    "VOLC": 1,     # Volcanics -> Granite (basement)
    "QGVOLC": 1,   # QG Volcanics -> Granite
}


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


# ─── Data preparation ───────────────────────────────────────────────────────

def load_drillholes():
    """Load real drill holes from extracted JSON."""
    path = os.path.join(OUTPUT_DIR, "real_drillholes.json")
    with open(path) as f:
        wells = json.load(f)
    print(f"Loaded {len(wells)} real drill holes")
    return wells


def well_to_training_points(well, coord_min, coord_max, sample_spacing=20.0):
    """Convert a well's geology intervals to (x,y,z) -> label training points.

    Samples along the well's depth at given spacing (default 20m = voxel Z spacing).
    Depths are measured from collar, so z = collar_z - depth.
    """
    coords = []
    labels = []

    for interval in well["intervals"]:
        fm = interval["formation"]
        code = FM_TO_CODE.get(fm)
        if code is None:
            continue

        from_depth = interval["from_depth"]
        to_depth = interval["to_depth"]

        # Sample points along the interval
        n_samples = max(1, int((to_depth - from_depth) / sample_spacing))
        depths = np.linspace(from_depth, to_depth, n_samples, endpoint=False)

        for d in depths:
            z = well["z_collar"] - d  # Convert depth to elevation
            coords.append([well["x"], well["y"], z])
            labels.append(code - 1)  # 0-indexed

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int64)


def prepare_data(wells):
    """Split wells into train/test and build training arrays."""
    # Shuffle with fixed seed
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(wells))

    n_test = 20
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_wells = [wells[i] for i in train_idx]
    test_wells = [wells[i] for i in test_idx]

    print(f"Train wells: {len(train_wells)}")
    print(f"Test wells:  {len(test_wells)}")

    # Compute normalization bounds (same as original for compatibility)
    coord_min = np.array([
        ORIGIN[0],
        ORIGIN[1] + (NY - 1) * SPACING[1],
        ORIGIN[2] + (NZ - 1) * SPACING[2],
    ])
    coord_max = np.array([
        ORIGIN[0] + (NX - 1) * SPACING[0],
        ORIGIN[1],
        ORIGIN[2],
    ])

    # Build training points
    all_coords = []
    all_labels = []
    for w in train_wells:
        c, l = well_to_training_points(w, coord_min, coord_max)
        if len(c) > 0:
            all_coords.append(c)
            all_labels.append(l)

    train_coords = np.vstack(all_coords)
    train_labels = np.concatenate(all_labels)

    # Build test points
    test_coords_list = []
    test_labels_list = []
    for w in test_wells:
        c, l = well_to_training_points(w, coord_min, coord_max)
        if len(c) > 0:
            test_coords_list.append(c)
            test_labels_list.append(l)

    test_coords = np.vstack(test_coords_list)
    test_labels = np.concatenate(test_labels_list)

    print(f"Training points: {len(train_labels):,}")
    print(f"Test points:     {len(test_labels):,}")

    # Normalize to [-1, 1]
    rng_coord = coord_max - coord_min

    def normalize(coords):
        normed = np.empty_like(coords)
        normed[:, 0] = 2.0 * (coords[:, 0] - coord_min[0]) / rng_coord[0] - 1.0
        normed[:, 1] = 2.0 * (coords[:, 1] - coord_min[1]) / rng_coord[1] - 1.0
        normed[:, 2] = 2.0 * (coords[:, 2] - coord_min[2]) / rng_coord[2] - 1.0
        return normed

    train_normed = normalize(train_coords)
    test_normed = normalize(test_coords)

    # Class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"\nTraining class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {LITHOLOGY_MAP[u+1]:<12} {c:>5} ({c/len(train_labels)*100:.1f}%)")

    # Class weights
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(unique)

    # Save metadata
    norm_params = {"coord_min": coord_min.tolist(), "coord_max": coord_max.tolist()}
    with open(os.path.join(MODEL_DIR, "norm_params.json"), "w") as f:
        json.dump(norm_params, f)

    split_info = {
        "n_train_wells": len(train_wells),
        "n_test_wells": len(test_wells),
        "train_well_names": [w["name"] for w in train_wells],
        "test_well_names": [w["name"] for w in test_wells],
        "n_train_points": int(len(train_labels)),
        "n_test_points": int(len(test_labels)),
    }
    with open(os.path.join(MODEL_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Build full weight tensor (for all 13 classes, some may be missing)
    full_weights = torch.ones(13, dtype=torch.float32)
    for u, w in zip(unique, weights):
        full_weights[u] = w

    return (train_normed, train_labels, test_normed, test_labels,
            test_wells, coord_min, coord_max, full_weights)


# ─── Training ───────────────────────────────────────────────────────────────

def train_model(model, train_coords, train_labels, test_coords, test_labels,
                class_weights, epochs=200, batch_size=4096, lr=1e-3):
    """Train on sparse drill hole data with data augmentation."""
    model = model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Convert to tensors
    train_c = torch.from_numpy(train_coords).float()
    train_l = torch.from_numpy(train_labels).long()
    test_c = torch.from_numpy(test_coords).float().to(DEVICE)
    test_l = torch.from_numpy(test_labels).long().to(DEVICE)

    best_test_acc = 0
    history = {"train_loss": [], "test_loss": [], "test_acc": [], "per_class_acc": []}

    n_train = len(train_l)
    print(f"\nTraining on {DEVICE}")
    print(f"Points: {n_train:,} train, {len(test_l):,} test")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print("-" * 70)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        # Shuffle
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            coords = train_c[idx].to(DEVICE)
            labs = train_l[idx].to(DEVICE)

            # Data augmentation: small spatial jitter (simulate measurement noise)
            noise = torch.randn_like(coords) * 0.005  # small in normalized space
            coords = coords + noise

            logits = model(coords)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Test
        model.eval()
        with torch.no_grad():
            test_logits = model(test_c)
            test_loss = criterion(test_logits, test_l).item()
            preds = test_logits.argmax(dim=-1)
            correct = (preds == test_l).sum().item()
            test_acc = correct / len(test_l)

            # Per-class accuracy
            per_class = {}
            for c in range(13):
                m = test_l == c
                if m.sum() > 0:
                    class_acc = (preds[m] == c).sum().item() / m.sum().item()
                    per_class[LITHOLOGY_MAP[c + 1]] = round(class_acc, 4)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["per_class_acc"].append(per_class)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
                  f"Acc: {test_acc:.4f} | Best: {best_test_acc:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 50 == 0:
            for name, acc in sorted(per_class.items(), key=lambda x: -x[1]):
                bar = "█" * int(acc * 30)
                print(f"    {name:<12} {acc:.3f} {bar}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pt"))
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    return history


# ─── Inference ──────────────────────────────────────────────────────────────

def predict_volume(model):
    """Predict lithology over the full grid."""
    model = model.to(DEVICE)
    model.eval()

    with open(os.path.join(MODEL_DIR, "norm_params.json")) as f:
        norm = json.load(f)
    coord_min = np.array(norm["coord_min"])
    coord_max = np.array(norm["coord_max"])
    rng = coord_max - coord_min

    print(f"Predicting full volume ({NX}x{NY}x{NZ})...")
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


def evaluate_vs_ground_truth(predicted):
    """Compare predicted volume against full geological model."""
    gt_path = os.path.join(OUTPUT_DIR, "voxet_lithology.npy")
    if not os.path.exists(gt_path):
        print("Ground truth voxel model not found, skipping comparison.")
        return

    gt = np.load(gt_path)
    valid = gt != NODATA

    pred_valid = predicted[valid]
    gt_valid = gt[valid].astype(np.int8)

    overall_acc = (pred_valid == gt_valid).sum() / len(gt_valid)
    print(f"\nFull volume comparison (vs geological model):")
    print(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"  Valid voxels compared: {len(gt_valid):,}")

    per_class = {}
    for c in range(1, 14):
        m = gt_valid == c
        if m.sum() > 0:
            acc = (pred_valid[m] == c).sum() / m.sum()
            per_class[LITHOLOGY_MAP[c]] = {"accuracy": round(float(acc), 4), "count": int(m.sum())}
            print(f"  {LITHOLOGY_MAP[c]:<12} {acc:.3f} ({m.sum():>10,} voxels)")

    results = {
        "overall_accuracy": round(float(overall_acc), 4),
        "n_voxels": int(len(gt_valid)),
        "per_class": per_class,
    }
    with open(os.path.join(MODEL_DIR, "ground_truth_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("REALISTIC TRAINING: Only real drill hole data")
    print("=" * 70)

    wells = load_drillholes()

    (train_coords, train_labels, test_coords, test_labels,
     test_wells, coord_min, coord_max, class_weights) = prepare_data(wells)

    # Bigger network + more epochs to compensate for sparse data
    model = NeuralField(n_classes=13, n_freqs=10, hidden_dim=256, n_layers=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    history = train_model(
        model, train_coords, train_labels, test_coords, test_labels,
        class_weights,
        epochs=200,
        batch_size=4096,
        lr=1e-3,
    )

    print("\n--- Full volume inference ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                                     map_location=DEVICE, weights_only=True))
    predicted = predict_volume(model)
    evaluate_vs_ground_truth(predicted)
    print("\nDone!")
