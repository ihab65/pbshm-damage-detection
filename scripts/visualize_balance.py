"""
Visualize damage-severity balance across Zone 1 / Zone 2 / Zone 3.

Generates four panels:
  1-3. Per-zone histograms (one per zone)
    4. Overlay of all three zones for direct comparison

Also prints a quick balance summary to the console.

Usage:
    uv run scripts/visualize_balance.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths (relative to this script, CWD-agnostic) ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
DATASET_PATH = os.path.join(DATA_DIR, "FINAL_SHM_DATASET_5000.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
sev_cols = ["Zone1_Sev", "Zone2_Sev", "Zone3_Sev"]
zone_labels = ["Zone 1", "Zone 2", "Zone 3"]
colors = ["#6366f1", "#f59e0b", "#10b981"]  # indigo, amber, emerald

# ── severity bins (align edges to the 0.05 grid) ───────────────────────────
bins = np.arange(-0.025, 0.90, 0.05)  # centres land on 0.00, 0.05, …, 0.85

# ── console summary ────────────────────────────────────────────────────────
print(f"Dataset: {len(df)} scenarios\n")
for col, label in zip(sev_cols, zone_labels):
    counts = df[col].value_counts().sort_index()
    min_c, max_c = counts.min(), counts.max()
    imbalance = max_c / min_c if min_c > 0 else float("inf")
    print(f"  {label}:  {len(counts)} levels | "
          f"min count = {min_c} | max count = {max_c} | "
          f"imbalance ratio = {imbalance:.2f}")
print()

# ── figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#1e1e2e")

for idx, (ax, col, label, color) in enumerate(
    zip(axes.flat[:3], sev_cols, zone_labels, colors)
):
    ax.set_facecolor("#2a2a3d")
    counts, edges, bars = ax.hist(
        df[col], bins=bins, color=color, edgecolor="#1e1e2e",
        linewidth=0.8, alpha=0.9
    )
    # annotate bar counts
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, c + max(counts) * 0.02,
                f"{int(c)}", ha="center", va="bottom",
                fontsize=7, color="white", fontweight="bold"
            )
    ax.set_title(label, fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xlabel("Damage severity", fontsize=10, color="#ccc")
    ax.set_ylabel("Count", fontsize=10, color="#ccc")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#444")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    # ideal uniform line
    ideal = len(df) / len(np.arange(0, 0.90, 0.05))
    ax.axhline(ideal, color="#ff6b6b", ls="--", lw=1.2, label=f"Ideal ({ideal:.0f})")
    ax.legend(fontsize=8, facecolor="#2a2a3d", edgecolor="#555",
              labelcolor="white", loc="upper right")

# panel 4 — overlay comparison
ax_overlay = axes[1, 1]
ax_overlay.set_facecolor("#2a2a3d")
for col, label, color in zip(sev_cols, zone_labels, colors):
    ax_overlay.hist(
        df[col], bins=bins, color=color, edgecolor="#1e1e2e",
        linewidth=0.6, alpha=0.55, label=label
    )
ideal = len(df) / len(np.arange(0, 0.90, 0.05))
ax_overlay.axhline(ideal, color="#ff6b6b", ls="--", lw=1.2, label=f"Ideal ({ideal:.0f})")
ax_overlay.set_title("All Zones (overlay)", fontsize=14, fontweight="bold",
                     color="white", pad=10)
ax_overlay.set_xlabel("Damage severity", fontsize=10, color="#ccc")
ax_overlay.set_ylabel("Count", fontsize=10, color="#ccc")
ax_overlay.tick_params(colors="#aaa")
ax_overlay.spines[:].set_color("#444")
ax_overlay.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax_overlay.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax_overlay.legend(fontsize=8, facecolor="#2a2a3d", edgecolor="#555",
                  labelcolor="white", loc="upper right")

fig.suptitle("Damage Severity Distribution per Zone",
             fontsize=18, fontweight="bold", color="white", y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUTPUT_DIR, "severity_balance.png")
fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

print(f"[OK] Figure saved to: {out_path}")
