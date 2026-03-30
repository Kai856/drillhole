"""Benchmark: evaluate trained models against ground truth.

Compares both the full-voxet model and the realistic (drill-hole-only) model,
producing per-class accuracy tables, confusion matrices, and summary metrics.

Usage:
    python pipeline/benchmark.py                    # benchmark both models
    python pipeline/benchmark.py --model synthetic   # only full-voxet model
    python pipeline/benchmark.py --model realistic   # only drill-hole model
"""
import numpy as np
import torch
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from config import NX, NY, NZ, ORIGIN, SPACING, NODATA, LITHOLOGY_MAP, OUTPUT_DIR
from neural_field_classes import NeuralField

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_ground_truth():
    path = os.path.join(OUTPUT_DIR, "voxet_lithology.npy")
    if not os.path.exists(path):
        print(f"Ground truth not found: {path}")
        print("Run pipeline/01_load_voxet.py first to extract voxet data.")
        sys.exit(1)
    return np.load(path)


def load_model(model_dir, n_layers):
    model = NeuralField(n_classes=13, n_freqs=10, hidden_dim=256, n_layers=n_layers)
    weights_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(weights_path):
        print(f"Model weights not found: {weights_path}")
        return None
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_predictions(model_dir):
    path = os.path.join(model_dir, "predicted_volume.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_metrics(gt, pred, valid_mask):
    """Compute per-class and overall accuracy metrics."""
    gt_valid = gt[valid_mask].astype(int)
    pred_valid = pred[valid_mask].astype(int)

    overall_acc = (gt_valid == pred_valid).sum() / len(gt_valid)

    results = {
        "overall_accuracy": round(float(overall_acc), 4),
        "n_voxels_evaluated": int(len(gt_valid)),
        "per_class": {},
    }

    # Per-class metrics
    for code in range(1, 14):
        mask = gt_valid == code
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((pred_valid[mask] == code).sum())
        acc = correct / n
        results["per_class"][LITHOLOGY_MAP[code]] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": n,
            "fraction_of_volume": round(n / len(gt_valid), 4),
        }

    return results


def print_metrics(results, title):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Overall accuracy: {results['overall_accuracy']:.4f} "
          f"({results['overall_accuracy']*100:.1f}%)")
    print(f"  Voxels evaluated: {results['n_voxels_evaluated']:,}")
    print()
    print(f"  {'Formation':<12} {'Accuracy':>10} {'Correct':>10} "
          f"{'Total':>12} {'Volume %':>10}")
    print(f"  {'-'*56}")

    for name, m in sorted(results["per_class"].items(),
                          key=lambda x: -x[1]["accuracy"]):
        bar = "█" * int(m["accuracy"] * 20)
        print(f"  {name:<12} {m['accuracy']:>9.4f} {m['correct']:>10,} "
              f"{m['total']:>12,} {m['fraction_of_volume']*100:>9.1f}%  {bar}")
    print()


# ─── Benchmark: Synthetic (full-voxet training) ────────────────────────────

def benchmark_synthetic(gt):
    """Benchmark the model trained on the full voxet grid."""
    model_dir = os.path.join(OUTPUT_DIR, "neural_field")
    print("\n--- Synthetic Model (full voxet training, 5-layer) ---")

    pred = load_predictions(model_dir)
    if pred is None:
        print(f"  No predictions found at {model_dir}/predicted_volume.npy")
        print("  Run pipeline/06_neural_field.py first.")
        return None

    valid = gt != NODATA

    # Full volume benchmark
    results = compute_metrics(gt, pred, valid)
    print_metrics(results, "SYNTHETIC MODEL — Full Volume Accuracy")

    # Held-out wells benchmark
    split_path = os.path.join(model_dir, "split_info.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)

        well_mask_2d = np.zeros((NX, NY), dtype=bool)
        radius = split["well_radius_cells"]
        for ci, cj in zip(split["well_centers_i"], split["well_centers_j"]):
            well_mask_2d[ci-radius:ci+radius, cj-radius:cj+radius] = True
        well_mask_3d = np.repeat(well_mask_2d[:, :, np.newaxis], NZ, axis=2)
        well_valid = valid & well_mask_3d

        well_results = compute_metrics(gt, pred, well_valid)
        print_metrics(well_results,
                      f"SYNTHETIC MODEL — Held-Out Wells Only "
                      f"({split['n_wells']} wells, "
                      f"{split['well_block_size_km']:.0f}km blocks)")

        results["held_out_wells"] = well_results

    # Save
    out_path = os.path.join(model_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


# ─── Benchmark: Realistic (drill-hole-only training) ──────────────────────

def benchmark_realistic(gt):
    """Benchmark the model trained only on real drill hole data."""
    model_dir = os.path.join(OUTPUT_DIR, "realistic_model")
    print("\n--- Realistic Model (86 real drill holes, 8-layer) ---")

    pred = load_predictions(model_dir)
    if pred is None:
        print(f"  No predictions found at {model_dir}/predicted_volume.npy")
        print("  Run pipeline/09_realistic_training.py first.")
        return None

    valid = gt != NODATA
    results = compute_metrics(gt, pred, valid)
    print_metrics(results, "REALISTIC MODEL — Full Volume vs Ground Truth")

    # Test well accuracy from training history
    split_path = os.path.join(model_dir, "split_info.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        results["train_wells"] = split["n_train_wells"]
        results["test_wells"] = split["n_test_wells"]
        results["test_well_names"] = split.get("test_well_names", [])

    history_path = os.path.join(model_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        best_epoch = int(np.argmax(history["test_acc"]))
        results["best_test_well_accuracy"] = round(history["test_acc"][best_epoch], 4)
        results["best_epoch"] = best_epoch + 1
        results["final_test_well_accuracy"] = round(history["test_acc"][-1], 4)
        print(f"  Test well accuracy: {results['best_test_well_accuracy']:.4f} "
              f"(epoch {results['best_epoch']})")

    # Save
    out_path = os.path.join(model_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


# ─── Side-by-side comparison ──────────────────────────────────────────────

def compare_models(synthetic_results, realistic_results):
    """Print side-by-side comparison of both models."""
    if not synthetic_results or not realistic_results:
        return

    print(f"\n{'=' * 70}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30} {'Synthetic':>15} {'Realistic':>15}")
    print(f"  {'-'*62}")

    s_acc = synthetic_results["overall_accuracy"]
    r_acc = realistic_results["overall_accuracy"]
    print(f"  {'Overall accuracy':<30} {s_acc:>14.1%} {r_acc:>14.1%}")

    # Per-class comparison
    all_classes = set(synthetic_results["per_class"].keys()) | \
                  set(realistic_results["per_class"].keys())
    for name in sorted(all_classes):
        s = synthetic_results["per_class"].get(name, {}).get("accuracy", 0)
        r = realistic_results["per_class"].get(name, {}).get("accuracy", 0)
        print(f"  {name:<30} {s:>14.1%} {r:>14.1%}")

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trained models")
    parser.add_argument("--model", choices=["synthetic", "realistic", "both"],
                        default="both", help="Which model to benchmark")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    gt = load_ground_truth()
    print(f"Ground truth volume: {gt.shape}")

    synthetic_results = None
    realistic_results = None

    if args.model in ("synthetic", "both"):
        synthetic_results = benchmark_synthetic(gt)

    if args.model in ("realistic", "both"):
        realistic_results = benchmark_realistic(gt)

    if args.model == "both":
        compare_models(synthetic_results, realistic_results)

    print("Benchmark complete.")
