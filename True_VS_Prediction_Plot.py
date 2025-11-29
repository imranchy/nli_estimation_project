import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ===========================================================
# GLOBAL STYLE MATCHED TO PREVIOUS FIGURES
# ===========================================================
sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})

BOLD = {"fontweight": "bold"}

# ===========================================================
# LOAD THE FOUR PREDICTION FILES
# ===========================================================
datasets = {
    "Cij-SCI":  pd.read_csv("PhysicsXGB_preds_cijs_sci.csv"),
    "Cij-XCI":  pd.read_csv("PhysicsXGB_preds_cijs_xci.csv"),
    "rho-SCI":  pd.read_csv("PhysicsXGB_preds_rho_sci.csv"),
    "rho-XCI":  pd.read_csv("PhysicsXGB_preds_rho_xci.csv"),
}

# ===========================================================
# CREATE MASTER 2×2 FIGURE WITH HISTOGRAMS
# ===========================================================
fig = plt.figure(figsize=(24, 16))
gs = GridSpec(2, 4, width_ratios=[3.6, 1.2, 3.6, 1.2], hspace=0.32, wspace=0.25)

axes_scatter = []
axes_hist = []

# allocate plot positions
for row in range(2):
    for col in [0, 2]:
        axes_scatter.append(fig.add_subplot(gs[row, col]))
        axes_hist.append(fig.add_subplot(gs[row, col + 1]))

# compute residual limits globally for color scale
all_residuals = []
for df in datasets.values():
    df["Residual"] = df["Pred"] - df["True"]
    all_residuals.extend(df["Residual"])

vmin = np.percentile(all_residuals, 2)
vmax = np.percentile(all_residuals, 98)

# ===========================================================
# PLOT EACH DATASET
# ===========================================================
for (name, df), ax_s, ax_h in zip(datasets.items(), axes_scatter, axes_hist):

    df["Residual"] = df["Pred"] - df["True"]

    # KDE BACKGROUND
    sns.kdeplot(
        x=df["True"], y=df["Pred"],
        fill=True, cmap="Blues", alpha=0.28,
        thresh=0.04, levels=60, ax=ax_s
    )

    # SCATTER COLORED BY RESIDUAL
    sc = ax_s.scatter(
        df["True"], df["Pred"],
        c=df["Residual"], cmap="vlag",
        vmin=vmin, vmax=vmax,
        s=28, alpha=0.85
    )

    # DIAGONAL REFERENCE LINE
    tmin = min(df["True"].min(), df["Pred"].min())
    tmax = max(df["True"].max(), df["Pred"].max())
    ax_s.plot([tmin, tmax], [tmin, tmax], "k--", linewidth=2)

    # METRICS
    mse = np.mean((df["Pred"] - df["True"])**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((df["True"] - df["Pred"])**2)
    ss_tot = np.sum((df["True"] - np.mean(df["True"]))**2)
    r2 = 1 - ss_res/ss_tot

    # TITLE (Bold with big font)
    ax_s.set_title(
        f"{name}\nRMSE = {rmse:.4f}   |   R² = {r2:.3f}",
        fontsize=20, fontweight="bold"
    )

    ax_s.set_xlabel("True Value", fontsize=18, fontweight="bold")
    ax_s.set_ylabel("Predicted Value", fontsize=18, fontweight="bold")
    ax_s.grid(True, linestyle="--", alpha=0.4, linewidth=1.3)

    # ---------------------------------------------------
    # RESIDUAL HISTOGRAM
    # ---------------------------------------------------
    ax_h.hist(
        df["Residual"], bins=40, orientation="horizontal",
        color="firebrick", alpha=0.65
    )

    ax_h.axhline(0, color="k", linestyle="--", linewidth=2)
    ax_h.set_title("Residuals", fontsize=18, fontweight="bold")
    ax_h.set_xticks([])
    ax_h.set_yticklabels([])
    ax_h.grid(False)

# ===========================================================
# GLOBAL COLORBAR — Right Side
# ===========================================================
cb_ax = fig.add_axes([0.93, 0.20, 0.015, 0.60])
cbar = fig.colorbar(sc, cax=cb_ax)
cbar.ax.set_title("Residual\n(Pred - True)", fontsize=16, fontweight="bold")

# ===========================================================
# SAVE & SHOW
# ===========================================================
plt.savefig("PhysicsXGB_TruePred_with_Histograms.png", dpi=420, bbox_inches="tight")
plt.show()

print("Saved: PhysicsXGB_TruePred_with_Histograms.png")
