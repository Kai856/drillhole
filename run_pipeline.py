#!/usr/bin/env python3
"""Adavale Basin 3D Geological Model — Pipeline Runner.

Run the full pipeline or individual stages:

    python run_pipeline.py                  # run everything
    python run_pipeline.py load             # 1. load input data
    python run_pipeline.py train            # 2. train models
    python run_pipeline.py train synthetic  # 2a. train full-voxet model only
    python run_pipeline.py train realistic  # 2b. train drill-hole model only
    python run_pipeline.py benchmark        # 3. evaluate models vs ground truth
    python run_pipeline.py visualize        # 4. generate plots & comparisons
    python run_pipeline.py export           # 5. export VTI, STL, web assets

Prerequisites:
    - Input data in: Adavale 3D Geological Model/
    - Python packages: numpy, torch, matplotlib, pyvista, plotly, pandas
"""
import subprocess
import sys
import os

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")

STAGES = {
    "load": {
        "description": "Load raw voxet + formation masks",
        "scripts": ["01_load_voxet.py", "02_load_masks.py"],
    },
    "train": {
        "description": "Train neural field models",
        "scripts": ["06_neural_field.py", "09_realistic_training.py"],
        "substages": {
            "synthetic": {
                "description": "Train on full voxet grid (30 epochs)",
                "scripts": ["06_neural_field.py"],
            },
            "realistic": {
                "description": "Train on real drill hole data only (200 epochs)",
                "scripts": ["09_realistic_training.py"],
            },
        },
    },
    "benchmark": {
        "description": "Evaluate models vs ground truth",
        "scripts": ["benchmark.py"],
    },
    "visualize": {
        "description": "Generate 2D/3D visualizations + comparisons",
        "scripts": ["03_visualize.py", "07_compare_wells.py"],
    },
    "export": {
        "description": "Export VTI, STL, web viewer, probabilities",
        "scripts": ["04_export.py", "05_interactive_web.py",
                     "08_3d_viewer.py", "10_export_probabilities.py"],
    },
}


def run_script(script_name):
    path = os.path.join(PIPELINE_DIR, script_name)
    print(f"\n{'─' * 70}")
    print(f"  Running: {script_name}")
    print(f"{'─' * 70}\n")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\n  ERROR: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def run_stage(stage_name, substage=None):
    stage = STAGES[stage_name]

    if substage and "substages" in stage:
        sub = stage["substages"][substage]
        print(f"\n{'=' * 70}")
        print(f"  STAGE: {stage_name} / {substage} — {sub['description']}")
        print(f"{'=' * 70}")
        for script in sub["scripts"]:
            run_script(script)
    else:
        print(f"\n{'=' * 70}")
        print(f"  STAGE: {stage_name} — {stage['description']}")
        print(f"{'=' * 70}")
        for script in stage["scripts"]:
            run_script(script)


def print_usage():
    print(__doc__)
    print("Available stages:")
    for name, stage in STAGES.items():
        print(f"  {name:<15} {stage['description']}")
        if "substages" in stage:
            for sub_name, sub in stage["substages"].items():
                print(f"    {sub_name:<13} {sub['description']}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Run everything in order
        for stage_name in ["load", "train", "benchmark", "visualize", "export"]:
            run_stage(stage_name)
    elif args[0] in ("--help", "-h"):
        print_usage()
    elif args[0] in STAGES:
        substage = args[1] if len(args) > 1 else None
        run_stage(args[0], substage)
    else:
        print(f"Unknown stage: {args[0]}")
        print_usage()
        sys.exit(1)
