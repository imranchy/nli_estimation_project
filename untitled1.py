import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sys

from physics_model import PhysicsXGBRegressor
sys.modules["__main__"].PhysicsXGBRegressor = PhysicsXGBRegressor

# ===========================================================
# GLOBAL PLOTTING STYLE (MATCHES EARLIER FIGURES)
# ===========================================================
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})
BOLD = {"fontweight": "bold"}

palette = sns.color_palette("Blues", 5)

# ===========================================================
# CLEAN LABEL TRANSLATION
# ===========================================================
feature_symbols = {
    "phy_alpha_mean": "Avg. loss",
    "phy_alpha_absdiff": "Loss diff",
    "phy_alpha_prod": "Loss product",
    "phy_D_mean": "Avg. dispersion",
    "phy_D_absdiff": "Dispersion diff",
    "phy_D_prod": "Dispersion product",
    "phy_D_acc_mean": "Avg. disp. accum.",
    "phy_D_acc_absdiff": "Disp. accum. diff",
    "phy_D_acc_prod": "Disp. accum. product",
    "phy_D_acc_pre_mean": "Avg. pre-disp.",
    "phy_D_acc_pre_absdiff": "Pre-disp. diff",
    "phy_D_acc_pre_prod": "Pre-disp. product",
    "phy_D_acc_span_mean": "Avg. span-disp.",
    "phy_D_acc_span_absdiff": "Span-disp. diff",
    "phy_D_acc_span_prod": "Span-disp. product",
    "phy_effective_length_mean": "Avg. eff. length",
    "phy_effective_length_absdiff": "Eff. length diff",
    "phy_effective_length_prod": "Eff. length product",
    "phy_length_mean": "Avg. fiber length",
    "phy_length_absdiff": "Length diff",
    "phy_walkoff_abs": "Walk-off",
    "phy_walkoff_weighted": "Walk-off × Δf²",
    "phy_power_prod": "Power product",
    "phy_power_sum": "Power sum",
    "phy_gamma_effL": "Nonlinear length",
    "phy_Dacc_perL": "Dispersion / length",
    "phy_Dacc_abs": "Dispersion accum.",
    "phy_alphaL": "Loss × length",
    "phy_RsDf": "Spectral load × |Δf|",
}

def sym(f): return feature_symbols.get(f, f)

# ===========================================================
# MODEL & DATA PATHS
# ===========================================================
model_files = {
    "cij-sci": "PhysicsXGB_best_cij_sci.pkl",
    "cij-xci": "PhysicsXGB_best_cij_xci.pkl",
    "rho-sci": "PhysicsXGB_best_rho_sci.pkl",
    "rho-xci": "PhysicsXGB_best_rho_xci.pkl",
}

pred_files = {
    "cij-sci": "PhysicsXGB_preds_cijs_sci.csv",
    "cij-xci": "PhysicsXGB_preds_cijs_xci.csv",
    "rho-sci": "PhysicsXGB_preds_rho_sci.csv",
    "rho-xci": "PhysicsXGB_preds_rho_xci.csv",
}

dataset_titles = {
    "cij-sci": "cij-sci",
    "cij-xci": "cij-xci",
    "rho-sci": "rho-sci",
    "rho-xci": "rho-xci",
}

# ===========================================================
# PROCESS DATASETS
# ===========================================================
feature_plots = []
calibration_data = []

for name in model_files.keys():
    print("Processing:", name)

    model = joblib.load(model_files[name])
    booster = model.model.get_booster()

    gains = booster.get_score(importance_type="gain")
    feats = np.array(list(gains.keys()))
    vals = np.array(list(gains.values()))

    mask = np.array([f.startswith("phy_") for f in feats])
    feats = feats[mask]
    vals = vals[mask]

    idx = np.argsort(vals)[::-1][:3]
    feature_plots.append((dataset_titles[name], feats[idx], vals[idx]))

    dfp = pd.read_csv(pred_files[name])
    calibration_data.append((dataset_titles[name], dfp))

# ===========================================================
# BUILD FIGURE
# ===========================================================
fig = plt.figure(figsize=(38, 20))
outer = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.28)

# -----------------------------------------------------------
# ROW 1 — FEATURE IMPORTANCE (Top 3)
# -----------------------------------------------------------
for i, (title, feats, vals) in enumerate(feature_plots):
    ax = fig.add_subplot(outer[0, i])

    ax.barh(
        [sym(f) for f in feats[::-1]],
        vals[::-1],
        color=palette[1:4],
        edgecolor="black",
        linewidth=1.3
    )

    ax.margins(x=0.25)
    ax.set_title(title, fontsize=22, fontweight="bold")

    ax.tick_params(axis='y', labelsize=18, width=1.4)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.tick_params(axis='x', labelsize=18, width=1.4)

# -----------------------------------------------------------
# ROW 2 — CALIBRATION PLOTS
# -----------------------------------------------------------
for i, (title, df) in enumerate(calibration_data):
    ax = fig.add_subplot(outer[1, i])

    true_vals = df["True"].values
    pred_vals = df["Pred"].values

    bins = np.linspace(pred_vals.min(), pred_vals.max(), 25)
    bin_ids = np.digitize(pred_vals, bins)

    mean_pred, mean_true = [], []
    for b in range(1, len(bins)):
        mask = bin_ids == b
        if mask.sum() > 6:
            mean_pred.append(pred_vals[mask].mean())
            mean_true.append(true_vals[mask].mean())

    ax.plot(
        mean_pred, mean_true,
        marker='o', markersize=7,
        linewidth=2.6,
        color="#a10a00",
        label="Model Calibration"
    )

    dmin = min(true_vals.min(), pred_vals.min())
    dmax = max(true_vals.max(), pred_vals.max())
    ax.plot([dmin, dmax], [dmin, dmax],
            'k--', linewidth=2.0, label="Ideal")

    ax.set_title(f"{title} Calibration", fontsize=22, fontweight="bold")
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel("Predicted Value", fontsize=20, fontweight="bold")
    ax.set_ylabel("Mean True Value", fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(fontsize=17)

# -----------------------------------------------------------
# SAVE
# -----------------------------------------------------------
plt.tight_layout()
plt.savefig("PhysicsXGB_feature_importance.png", dpi=420, bbox_inches="tight")
plt.show()

print("\nSaved: PhysicsXGB_feature_importance.png")
