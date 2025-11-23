import pandas as pd
import matplotlib.pyplot as plt

# Load results summary file
df = pd.read_csv("all_results_summary_v20.csv")

# Remove tuning-only row
df = df[df["Model"] != "PhysicsXGB (tuning_val)"].copy()

# Convert R2 to %
df["R2_percent"] = df["R2"] * 100

# Get unique datasets
datasets = df["Dataset"].unique()

# Loop through datasets and generate one plot each
for ds in datasets:
    df_ds = df[df["Dataset"] == ds]

    plt.figure(figsize=(10,6))
    plt.bar(df_ds["Model"], df_ds["R2_percent"])

    plt.title(f"Test R² (%) — {ds}")
    plt.ylabel("Test R² (%)")
    plt.ylim(20, 100)         # Start axis at 20%
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
