import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_style("whitegrid")


# Load the four prediction files
datasets = {
    "Cij-SCI":  pd.read_csv("PhysicsXGB_preds_cijs_sci.csv"),
    "Cij-XCI":  pd.read_csv("PhysicsXGB_preds_cijs_xci.csv"),
    "rho-SCI":  pd.read_csv("PhysicsXGB_preds_rho_sci.csv"),
    "rho-XCI":  pd.read_csv("PhysicsXGB_preds_rho_xci.csv"),
}


# Create 2×2 grid, each with two subplots (scatter + histogram)
fig = plt.figure(figsize=(22, 14))
gs = GridSpec(2, 4, width_ratios=[3.5, 1.2, 3.5, 1.2], hspace=0.35, wspace=0.25)

axes_scatter = []
axes_hist = []

# Dataset loop
index = 0
all_residuals = []

for row in range(2):
    for col in [0, 2]:     # scatter positions
        axes_scatter.append(fig.add_subplot(gs[row, col]))
        axes_hist.append(fig.add_subplot(gs[row, col + 1]))
        
for df in datasets.values():
    df["Residual"] = df["Pred"] - df["True"]
    all_residuals.extend(df["Residual"])

# GLOBAL residual range for consistent color scale
vmin = np.percentile(all_residuals, 2)
vmax = np.percentile(all_residuals, 98)

# ======================================================================
# PLOT EACH DATASET
# ======================================================================

for (name, df), ax_s, ax_h in zip(datasets.items(), axes_scatter, axes_hist):

    df["Residual"] = df["Pred"] - df["True"]

    # ---- KDE Background ----
    sns.kdeplot(
        x=df["True"], y=df["Pred"],
        fill=True, thresh=0.05, cmap="Blues", alpha=0.32,
        levels=60, ax=ax_s
    )

    # ---- Scatter Colored by Residual ----
    sc = ax_s.scatter(
        df["True"], df["Pred"],
        c=df["Residual"], cmap="vlag",
        vmin=vmin, vmax=vmax,
        s=20, alpha=0.85, edgecolor="none"
    )

    # ---- Diagonal Line ----
    tmin = min(df["True"].min(), df["Pred"].min())
    tmax = max(df["True"].max(), df["Pred"].max())
    ax_s.plot([tmin, tmax], [tmin, tmax], "k--", linewidth=2)

    # ---- Stats ----
    mse = np.mean((df["Pred"] - df["True"])**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((df["True"] - df["Pred"])**2)
    ss_tot = np.sum((df["True"] - np.mean(df["True"]))**2)
    r2 = 1 - ss_res/ss_tot
    

    title = f"{name}\nRMSE = {rmse:.4f}   |   R² = {r2:.3f}"
    ax_s.set_title(title, fontsize=15, fontweight="bold")

    ax_s.set_xlabel("True Value", fontsize=13)
    ax_s.set_ylabel("Predicted Value", fontsize=13)

    # ---------------------------------------------------------------
    # HISTOGRAM (Residual distribution)
    # ---------------------------------------------------------------
    ax_h.hist(df["Residual"], bins=40, orientation="horizontal",
              color="firebrick", alpha=0.6, edgecolor="none")

    ax_h.axhline(0, color="k", linestyle="--", linewidth=2)  # zero-error line
    ax_h.set_title("Residuals", fontsize=12)
    #ax_h.set_xlabel("Count")
    ax_h.set_yticklabels([])

# ---------------------------------------------------------------
# GLOBAL COLORBAR — moved far right
# ---------------------------------------------------------------
cb_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cbar = fig.colorbar(sc, cax=cb_ax)
#cbar.ax.set_title("Residual\n(Pred - True)", fontsize=12)

plt.savefig("PhysicsXGB_TruePred_with_Histograms.png", dpi=400, bbox_inches="tight")
plt.show()

print("Saved: PhysicsXGB_TruePred_with_Histograms.png")
