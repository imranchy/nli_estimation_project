import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================
# 📌 Load CSV
# ============================================
df = pd.read_csv("summary.csv").copy()
df["Time_rounded"] = df["Time"].round(0).astype(int)

# ============================================
# 📌 Global Styling (All Bold + Larger Font)
# ============================================
BOLD = {'fontweight': 'bold'}
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})

model_names = list(df["Model"].unique())

# ============================================
# 📌 FIGURE 1: Scatter (Legend top-left inside)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# --------- Plot 1: R² ---------
sns.scatterplot(
    data=df, x="Time_rounded", y="R2",
    hue="Model", style="Dataset",
    s=220, palette="Set2", edgecolor="black", ax=axes[0]
)

axes[0].set_xlabel("Training Time (s)", **BOLD)
axes[0].set_ylabel("R² Score", **BOLD)
axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=1.4)

# --------- Plot 2: RMSE ---------
sns.scatterplot(
    data=df, x="Time_rounded", y="RMSE",
    hue="Model",
    s=220, palette="Set2", edgecolor="black", ax=axes[1]
)

axes[1].set_xlabel("Training Time (s)", **BOLD)
axes[1].set_ylabel("RMSE Score", **BOLD)
axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=1.4)

# ============================================
# 📌 COMPACT LEGEND INSIDE (Top-Left, 2-Column)
# ============================================
# Get only Model entries
all_handles, all_labels = axes[0].get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(all_handles, all_labels) if l in model_names]
handles, labels = zip(*filtered)

legend1 = fig.legend(
    handles, labels,
    title="Model",
    loc="upper left",
    bbox_to_anchor=(0.02, 1.02),
    ncol=2,
    fontsize=18,
    frameon=True
)
plt.setp(legend1.get_title(), fontweight='bold', fontsize=18)

# Remove seaborn's internal legends
for ax in axes:
    lg = ax.get_legend()
    if lg:
        lg.remove()

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("joint_time_vs_metrics_scatter.png", dpi=350)
plt.show()

print("Saved: joint_time_vs_metrics_scatter.png")


# ============================================
# 📌 FIGURE 2: Bar Chart Grid (Legend top-left inside)
# ============================================
metrics = ["MSE", "RMSE", "R2", "Time"]
titles = ["MSE Score", "RMSE Score", "R² Score", "Training Time (s)"]

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for ax, metric, title in zip(axes.flatten(), metrics, titles):

    sns.barplot(
        data=df, x="Dataset", y=metric,
        hue="Model", palette="Set2", ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel(title, **BOLD)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# Legend filtering
all_handles, all_labels = axes[0][0].get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(all_handles, all_labels) if l in model_names]
handles, labels = zip(*filtered)

legend2 = fig.legend(
    handles, labels,
    title="Model",
    loc="upper left",
    bbox_to_anchor=(0.02, 1.02),
    ncol=2,
    fontsize=18,
    frameon=True
)
plt.setp(legend2.get_title(), fontweight='bold', fontsize=18)

# Remove internal legends
for ax in axes.flatten():
    if ax.get_legend() is not None:
        ax.get_legend().remove()

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("ccombined_per_dataset_metrics.png", dpi=350)
plt.show()

print("Saved: combined_per_dataset_metrics.png")
