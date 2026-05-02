import os
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(SCRIPT_DIR, "..", "data", "processed", "batches")
master_combo_file = os.path.join(SCRIPT_DIR, "..", "data", "processed", "master_damage_combinations.npy")

# 1. Explicitly grab and sort Phase 1 files
phase1_files = sorted(glob.glob(os.path.join(output_dir, "batch_0*.csv")))
print(f"Found {len(phase1_files)} Phase 1 batches (100-size).")

# 2. Explicitly grab and sort Phase 2 files
phase2_files = sorted(glob.glob(os.path.join(output_dir, "batch_part2_*.csv")))
print(f"Found {len(phase2_files)} Phase 2 batches (20-size).")

# 3. Concatenate them in the EXACT order they were generated
all_files_in_order = phase1_files + phase2_files

df_features = pd.concat((pd.read_csv(f) for f in all_files_in_order), ignore_index=True)
num_generated = len(df_features)

print(f"Successfully stitched {num_generated} total scenarios.")

# 4. Slice the master labels to match
labels = np.load(master_combo_file)
labels_sliced = labels[:num_generated]

df_labels = pd.DataFrame(labels_sliced, columns=["Zone1_Sev", "Zone2_Sev", "Zone3_Sev"])

# 5. Final Assembly
final_dataset = pd.concat([df_features, df_labels], axis=1)

out_path = os.path.join(SCRIPT_DIR, "..", "data", "processed", f"FINAL_SHM_DATASET_{num_generated}.csv")
final_dataset.to_csv(out_path, index=False)

print(f"[OK] Dataset saved to: {out_path}")