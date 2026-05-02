"""
continue_generation.py - Phase 2 of ETABS Data Generation
==========================================================
Continues the 5,000-scenario campaign using smaller batches (default 20)
and reboots ETABS between each batch to counteract memory-leak slowdown.

Usage
-----
Run this script from the project root:

    .venv\\Scripts\\python.exe scripts\\continue_generation.py

The script is FULLY AUTOMATED:
  - Launches a fresh ETABS process for each batch (no manual open needed)
  - Copies the master .EDB to a temp file (original is never modified)
  - Kills ETABS and cleans up after each batch
  - Resume-safe: if it crashes, just re-run and it picks up where it left off
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.etabs_api import run_batches_with_reboot

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
BATCH_SIZE = 20          # Smaller = ETABS stays in its fastest state
N_MODES = 4              # Optimal elbow point from convergence analysis

GROUP_NAME = "OPT_20"
MAT_NAMES = ["C30/37 Zone 1", "C30/37 Zone 2", "C30/37 Zone 3"]
E_I = 32836.6

# Paths (resolved relative to this script's location)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MASTER_MODEL_PATH = os.path.join(_ROOT, 'data', 'external', 'main_model.EDB')
MASTER_COMBO_FILE = os.path.join(_ROOT, 'data', 'processed',
                                 'master_damage_combinations.npy')
OUTPUT_DIR = os.path.join(_ROOT, 'data', 'processed', 'batches')
NOT_DAMAGED_FILE = os.path.join(_ROOT, 'data', 'processed',
                                'notDamaged_Flexibility.pkl')

COOLDOWN_SECONDS = 10    # Pause between ETABS restart cycles


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ETABS Data Generation - Reboot Mode")
    print("=" * 60)

    # Verify the master model exists before we start
    if not os.path.isfile(MASTER_MODEL_PATH):
        print(f"\n[ERROR] Master model not found:\n  {MASTER_MODEL_PATH}")
        print("Please verify the path in this script's configuration.")
        sys.exit(1)

    # Load master damage combinations
    dataset_array = np.load(MASTER_COMBO_FILE)
    print(f"\nMaster array: {dataset_array.shape[0]} total scenarios")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Cooldown:     {COOLDOWN_SECONDS}s between batches")
    print(f"Model:        {MASTER_MODEL_PATH}")

    # Run the reboot-enabled batch loop
    new_files = run_batches_with_reboot(
        dataset_array=dataset_array,
        batch_size=BATCH_SIZE,
        group_name=GROUP_NAME,
        E_i=E_I,
        mat_names=MAT_NAMES,
        not_damaged_file=NOT_DAMAGED_FILE,
        output_dir=OUTPUT_DIR,
        master_model_path=MASTER_MODEL_PATH,
        n_modes=N_MODES,
        cooldown=COOLDOWN_SECONDS,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! Generated {len(new_files)} new batch files.")
    print(f"  Run the Phase 3 assembly cell in the notebook to")
    print(f"  stitch everything into FINAL_SHM_DATASET.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
