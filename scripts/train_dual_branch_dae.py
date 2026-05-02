"""
train_dual_branch_dae.py -- Train the Dual-Branch Denoising Autoencoder
=======================================================================

This script implements the full two-phase training pipeline:

  Phase 1  --  Self-supervised denoising autoencoder
               - Input: 60-D flexibility indicator (20 sensors x 3 DOFs)
               - Random sensor masking simulates missing instrumentation
               - Loss: MSE(reconstruction, CLEAN original signal)
               - Explores multiple drop-off rates to find the sweet spot

  Phase 2  --  Joint dual-branch training
               - Encoder + Decoder + Predictor trained simultaneously
               - Combined loss: recon_weight * MSE_recon + pred_weight * MSE_sev
               - Forces latent space to encode severity-discriminative features

Run from the project root:
    uv run scripts/train_dual_branch_dae.py

The best models are saved under  data/processed/models/
Figures are saved under          data/processed/figures/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -- Make src/ importable ------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

from src.modeling import (
    build_denoising_autoencoder,
    build_dual_branch_model,
    build_encoder,
    INPUT_DIM,
    LATENT_DIM,
    OUTPUT_ZONES,
)

# -- Paths ---------------------------------------------------------------
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
DATASET     = os.path.join(DATA_DIR, "FINAL_SHM_DATASET_5000.csv")
MODEL_DIR   = os.path.join(DATA_DIR, "models")
FIGURE_DIR  = os.path.join(DATA_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# -- Reproducibility -----------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===================================================================
# 1. DATA LOADING & PREPARATION
# ===================================================================

print("=" * 65)
print(" LOADING DATASET")
print("=" * 65)

df = pd.read_csv(DATASET)
print(f"  Loaded {len(df)} scenarios, {len(df.columns)} columns.\n")

# Separate features (60 flexibility indicators) and labels (3 zone severities)
feature_cols = [str(i) for i in range(INPUT_DIM)]  # "0" .. "59"
label_cols   = ["Zone1_Sev", "Zone2_Sev", "Zone3_Sev"]
from sklearn.preprocessing import StandardScaler

X_raw = df[feature_cols].values.astype(np.float32)   # (5000, 60)
y_raw = df[label_cols].values.astype(np.float32)      # (5000, 3)

# -- Standardise features (zero-mean, unit-variance) ---------------------
# StandardScaler is critical here: the raw flexibility indicators are
# ~1e-9 with std ~1e-8.  MinMaxScaler would amplify floating-point noise
# and destroy the subtle damage signal.  StandardScaler preserves the
# relative variance that encodes structural damage information.
scaler = StandardScaler()
X_all = scaler.fit_transform(X_raw).astype(np.float32)

# Severities are already in [0, 0.85] -- no scaling needed,
# the predictor's sigmoid output naturally covers this range.
y_all = y_raw

# -- Train / validation / test split -------------------------------------
# 70 / 15 / 15 stratification not needed -- combinations are unique
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765,  # 0.1765 * 0.85 ~ 0.15
    random_state=SEED
)

print(f"  Train : {len(X_train)}")
print(f"  Val   : {len(X_val)}")
print(f"  Test  : {len(X_test)}")
print()

# ===================================================================
# 2. PHASE 1 -- DROP-RATE SWEEP  (denoising autoencoder)
# ===================================================================
#
# We train the DAE at several masking drop-off rates and compare
# reconstruction quality on the validation set.
#
# Why this matters:
#   - drop_rate ~ 0  -> perfect reconstruction but no robustness to
#     missing sensors on B-type structures.
#   - drop_rate too high -> the encoder can't recover the signal.
#   - The sweet spot balances denoising capacity with reconstruction
#     accuracy and will determine how aggressively we can reduce
#     sensors on B-type structures.
# ===================================================================

DROP_RATES     = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
DAE_EPOCHS     = 150
DAE_BATCH_SIZE = 64
DAE_LR         = 1e-3

sweep_results = {}       # {rate: {"train_loss": [...], "val_loss": [...]}}
best_val_loss = float("inf")
best_drop_rate = None

print("=" * 65)
print(" PHASE 1 -- DROP-RATE SWEEP  (Denoising Autoencoder)")
print("=" * 65)

for rate in DROP_RATES:
    print(f"\n{'-'*50}")
    print(f"  Training DAE with drop_rate = {rate:.0%}")
    print(f"{'-'*50}")

    # Build a fresh DAE for each rate
    dae, encoder, decoder = build_denoising_autoencoder(
        drop_rate=rate, input_dim=INPUT_DIM, latent_dim=LATENT_DIM
    )
    dae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DAE_LR),
        loss="mse",
    )

    # The target is always the CLEAN input (self-supervised denoising)
    history = dae.fit(
        X_train, X_train,            # input = clean, target = clean
        validation_data=(X_val, X_val),
        epochs=DAE_EPOCHS,
        batch_size=DAE_BATCH_SIZE,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=10, min_lr=1e-6, verbose=1
            ),
        ],
    )

    final_val = min(history.history["val_loss"])
    print(f"  >> Best val MSE = {final_val:.6f}")

    sweep_results[rate] = {
        "train_loss": history.history["loss"],
        "val_loss":   history.history["val_loss"],
        "best_val":   final_val,
    }

    # Track global best
    if final_val < best_val_loss:
        best_val_loss  = final_val
        best_drop_rate = rate
        best_encoder   = encoder
        best_decoder   = decoder
        best_dae       = dae

print(f"\n{'='*65}")
print(f"  BEST DROP RATE: {best_drop_rate:.0%}  (val MSE = {best_val_loss:.6f})")
print(f"{'='*65}\n")

# -- Save best DAE components (Phase 1 checkpoint) ----------------------
best_encoder.save(os.path.join(MODEL_DIR, "encoder_phase1.keras"))
best_decoder.save(os.path.join(MODEL_DIR, "decoder_phase1.keras"))
best_dae.save(os.path.join(MODEL_DIR, "dae_phase1.keras"))

# Save the scaler for later inference
scaler_params = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
    "var_": scaler.var_.tolist(),
}
with open(os.path.join(MODEL_DIR, "feature_scaler.json"), "w") as f:
    json.dump(scaler_params, f)

# -- Plot: Drop-rate sweep comparison -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.patch.set_facecolor("#1e1e2e")

# Panel 1: learning curves per drop rate
ax1 = axes[0]
ax1.set_facecolor("#2a2a3d")
cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(DROP_RATES)))

for color, rate in zip(cmap, DROP_RATES):
    epochs = range(1, len(sweep_results[rate]["val_loss"]) + 1)
    ax1.plot(epochs, sweep_results[rate]["val_loss"],
             color=color, lw=1.5, label=f"{rate:.0%}")

ax1.set_xlabel("Epoch", color="#ccc", fontsize=11)
ax1.set_ylabel("Validation MSE", color="#ccc", fontsize=11)
ax1.set_title("Phase 1: Validation Loss per Drop Rate",
              color="white", fontsize=13, fontweight="bold")
ax1.tick_params(colors="#aaa")
ax1.spines[:].set_color("#444")
ax1.legend(title="Drop rate", fontsize=9, title_fontsize=10,
           facecolor="#2a2a3d", edgecolor="#555", labelcolor="white")
ax1.set_yscale("log")

# Panel 2: best val MSE vs drop rate (bar chart)
ax2 = axes[1]
ax2.set_facecolor("#2a2a3d")
rates_list = list(sweep_results.keys())
vals_list  = [sweep_results[r]["best_val"] for r in rates_list]
bar_colors = ["#10b981" if r == best_drop_rate else "#6366f1"
              for r in rates_list]
bars = ax2.bar([f"{r:.0%}" for r in rates_list], vals_list,
               color=bar_colors, edgecolor="#1e1e2e", width=0.6)
for bar, v in zip(bars, vals_list):
    ax2.text(bar.get_x() + bar.get_width()/2, v + max(vals_list)*0.02,
             f"{v:.5f}", ha="center", va="bottom", fontsize=9,
             color="white", fontweight="bold")

ax2.set_xlabel("Drop Rate", color="#ccc", fontsize=11)
ax2.set_ylabel("Best Validation MSE", color="#ccc", fontsize=11)
ax2.set_title("Reconstruction Quality vs. Masking Rate",
              color="white", fontsize=13, fontweight="bold")
ax2.tick_params(colors="#aaa")
ax2.spines[:].set_color("#444")

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "phase1_droprate_sweep.png"),
            dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  [SAVED] phase1_droprate_sweep.png\n")


# ===================================================================
# 3. PHASE 2 -- JOINT DUAL-BRANCH TRAINING
# ===================================================================
#
# Instead of freezing the encoder and training only the predictor,
# we train ALL three components jointly with a combined loss:
#
#   total = recon_weight * MSE(recon, clean) + pred_weight * MSE(sev, true_sev)
#
# This forces the encoder's latent space to encode features that are
# good for BOTH reconstruction AND severity prediction.
#
# We initialise from the best Phase-1 encoder/decoder weights so the
# reconstruction branch starts pre-converged, and the predictor head
# trains from scratch on top.
# ===================================================================

PRED_EPOCHS     = 300
PRED_BATCH_SIZE = 64
PRED_LR         = 1e-3
RECON_WEIGHT    = 1.0     # keep reconstruction capability
PRED_WEIGHT     = 5.0     # push latent space toward severity info

print("=" * 65)
print(f" PHASE 2 -- DUAL-BRANCH JOINT TRAINING  (init from {best_drop_rate:.0%} DAE)")
print("=" * 65)

dual_model, predictor = build_dual_branch_model(
    encoder=best_encoder,
    decoder=best_decoder,
    predictor=None,            # fresh predictor head
    drop_rate=best_drop_rate,
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    output_zones=OUTPUT_ZONES,
)

dual_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PRED_LR),
    loss={"decoder": "mse", "predictor": "mse"},
    loss_weights={"decoder": RECON_WEIGHT, "predictor": PRED_WEIGHT},
    metrics={"predictor": ["mae"]},
)

dual_model.summary()

# Targets: reconstruction branch -> clean input, predictor -> labels
history_pred = dual_model.fit(
    X_train,
    {"decoder": X_train, "predictor": y_train},
    validation_data=(X_val, {"decoder": X_val, "predictor": y_val}),
    epochs=PRED_EPOCHS,
    batch_size=PRED_BATCH_SIZE,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_predictor_loss", patience=30,
            restore_best_weights=True, verbose=1, mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_predictor_loss", factor=0.5,
            patience=15, min_lr=1e-6, verbose=1, mode="min"
        ),
    ],
)

# -- Save models ---------------------------------------------------------
best_encoder.save(os.path.join(MODEL_DIR, "encoder_best.keras"))
best_decoder.save(os.path.join(MODEL_DIR, "decoder_best.keras"))
predictor.save(os.path.join(MODEL_DIR, "predictor_best.keras"))
dual_model.save(os.path.join(MODEL_DIR, "dual_model_best.keras"))

# -- Test-set evaluation -------------------------------------------------
print(f"\n{'='*65}")
print(" TEST-SET EVALUATION")
print(f"{'='*65}")

# Predict with the full dual-branch model
test_recon, test_sev_pred = dual_model.predict(X_test, verbose=0)

# Reconstruction quality
recon_mse = np.mean((X_test - test_recon) ** 2)
print(f"  Reconstruction MSE : {recon_mse:.6f}")

# Severity quality
sev_mse = np.mean((y_test - test_sev_pred) ** 2)
sev_mae = np.mean(np.abs(y_test - test_sev_pred))
print(f"  Severity MSE       : {sev_mse:.6f}")
print(f"  Severity MAE       : {sev_mae:.6f}")

# Per-zone MAE
for i, zone in enumerate(label_cols):
    zone_mae = np.mean(np.abs(y_test[:, i] - test_sev_pred[:, i]))
    print(f"  {zone} MAE     : {zone_mae:.4f}")

# -- Plot: Phase 2 training curves --------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#1e1e2e")

# Total loss
ax = axes[0]
ax.set_facecolor("#2a2a3d")
ax.plot(history_pred.history["loss"], color="#6366f1", lw=1.5, label="Train")
ax.plot(history_pred.history["val_loss"], color="#f59e0b", lw=1.5, label="Val")
ax.set_xlabel("Epoch", color="#ccc")
ax.set_ylabel("Total Loss", color="#ccc")
ax.set_title("Phase 2: Total Loss", color="white", fontweight="bold")
ax.tick_params(colors="#aaa")
ax.spines[:].set_color("#444")
ax.legend(facecolor="#2a2a3d", edgecolor="#555", labelcolor="white")

# Predictor loss
ax = axes[1]
ax.set_facecolor("#2a2a3d")
ax.plot(history_pred.history["predictor_loss"],
        color="#6366f1", lw=1.5, label="Train")
ax.plot(history_pred.history["val_predictor_loss"],
        color="#f59e0b", lw=1.5, label="Val")
ax.set_xlabel("Epoch", color="#ccc")
ax.set_ylabel("Predictor MSE", color="#ccc")
ax.set_title("Phase 2: Predictor Loss", color="white", fontweight="bold")
ax.tick_params(colors="#aaa")
ax.spines[:].set_color("#444")
ax.legend(facecolor="#2a2a3d", edgecolor="#555", labelcolor="white")

# Predictor MAE
ax = axes[2]
ax.set_facecolor("#2a2a3d")
ax.plot(history_pred.history["predictor_mae"],
        color="#6366f1", lw=1.5, label="Train")
ax.plot(history_pred.history["val_predictor_mae"],
        color="#f59e0b", lw=1.5, label="Val")
ax.set_xlabel("Epoch", color="#ccc")
ax.set_ylabel("MAE", color="#ccc")
ax.set_title("Phase 2: Predictor MAE", color="white", fontweight="bold")
ax.tick_params(colors="#aaa")
ax.spines[:].set_color("#444")
ax.legend(facecolor="#2a2a3d", edgecolor="#555", labelcolor="white")

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "phase2_predictor_training.png"),
            dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n  [SAVED] phase2_predictor_training.png")

# -- Plot: Predicted vs True scatter (one per zone) ---------------------
y_pred = test_sev_pred

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#1e1e2e")
colors = ["#6366f1", "#f59e0b", "#10b981"]

for i, (ax, zone, color) in enumerate(zip(axes, label_cols, colors)):
    ax.set_facecolor("#2a2a3d")
    ax.scatter(y_test[:, i], y_pred[:, i], s=12, alpha=0.5,
               color=color, edgecolor="none")
    ax.plot([0, 0.85], [0, 0.85], ls="--", color="#ff6b6b", lw=1.2,
            label="Ideal")
    zone_mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
    ax.set_title(f"{zone}  (MAE={zone_mae:.4f})",
                 color="white", fontweight="bold")
    ax.set_xlabel("True Severity", color="#ccc")
    ax.set_ylabel("Predicted Severity", color="#ccc")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#2a2a3d", edgecolor="#555", labelcolor="white",
              fontsize=9)
    ax.set_xlim(-0.05, 0.90)
    ax.set_ylim(-0.05, 0.90)

fig.suptitle("Test Set: Predicted vs. True Damage Severity",
             color="white", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, "phase2_pred_vs_true.png"),
            dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print(f"  [SAVED] phase2_pred_vs_true.png")

# -- Summary -------------------------------------------------------------
print(f"\n{'='*65}")
print(" DONE -- All models and figures saved.")
print(f"{'='*65}")
print(f"  Models  -> {MODEL_DIR}")
print(f"  Figures -> {FIGURE_DIR}")
print(f"  Best drop rate  : {best_drop_rate:.0%}")
print(f"  DAE val MSE     : {best_val_loss:.6f}")
print(f"  Severity test   : MSE={sev_mse:.6f}  MAE={sev_mae:.6f}")
print(f"  Recon test      : MSE={recon_mse:.6f}")
