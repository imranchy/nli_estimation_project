import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load summary
df = pd.read_csv("summary.csv").copy()

# Round time values (cleaner axes)
df["Time_rounded"] = df["Time"].round(0).astype(int)

# Paper-quality styling
sns.set_theme(style="whitegrid", font_scale=1.4)

# -------------------------------------------------
#   📌 1×2 Horizontal Combined Figure (Scatter Only)
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# -------------------------------------------------
#  Plot 1 — Training Time vs R²
# -------------------------------------------------
sns.scatterplot(
    data=df,
    x="Time_rounded",
    y="R2",
    hue="Model",
    style="Dataset",
    s=180,
    palette="Set2",
    edgecolor="black",
    ax=axes[0]
)

axes[0].set_title("Training Time vs R²", fontsize=18, weight="bold")
axes[0].set_xlabel("Training Time (s)", fontsize=15)
axes[0].set_ylabel("R² Score", fontsize=15)
axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=1.4)

# -------------------------------------------------
#  Plot 2 — Training Time vs RMSE
# -------------------------------------------------
sns.scatterplot(
    data=df,
    x="Time_rounded",
    y="RMSE",
    hue="Model",
    style="Dataset",
    s=180,
    palette="Set2",
    edgecolor="black",
    ax=axes[1]
)

axes[1].set_title("Training Time vs RMSE", fontsize=18, weight="bold")
axes[1].set_xlabel("Training Time (s)", fontsize=15)
axes[1].set_ylabel("RMSE", fontsize=15)
axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=1.4)

# Move legend outside
axes[1].legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Model")

plt.tight_layout()

# Save HQ figure
plt.savefig("joint_time_vs_metrics_scatter.png", dpi=350, bbox_inches="tight")
plt.show()

print("Saved: joint_time_vs_metrics_scatter.png")


# Model performance per dataset
df = pd.read_csv("summary.csv")

metrics = ["MSE", "RMSE", "R2", "Time"]
titles = ["MSE Score", "RMSE Score", "R² Score", "Training Time (s)"]

plt.figure(figsize=(16, 12))

for i, (metric, title) in enumerate(zip(metrics, titles), 1):
    plt.subplot(2, 2, i)
    sns.barplot(
        data=df, x="Dataset", y=metric, hue="Model", palette="Set2"
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 🚫 Removed value annotations

plt.suptitle("Model Performance per Dataset Across All Metrics", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("combined_per_dataset_metrics.png", dpi=300)
plt.show()

print("Saved:", "combined_per_dataset_metrics.png")




