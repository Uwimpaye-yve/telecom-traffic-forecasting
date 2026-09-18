import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Plot Spatial Traffic Distribution across all 10,000 cells
def plot_spatial_distribution():
    print("Generating Figure 1: Spatial Traffic Distribution...")
    spatial_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "spatial_grid_totals.parquet"))
    traffic = spatial_df[spatial_df["total_internet"] > 0]["total_internet"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(traffic, bins=60, kde=True, log_scale=True, ax=ax, color="#1f77b4")
    ax.set_title("Distribution of Total Internet Traffic Across Milan Grid Cells (Log Scale)", fontsize=12)
    ax.set_xlabel("Total Internet Traffic Activity (Log Scale)", fontsize=10)
    ax.set_ylabel("Number of Grid Cells", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(FIGURES_DIR, "fig1_spatial_distribution.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# 2. Plot First 2 Weeks for the 5 Key Geographical Areas
def plot_two_week_comparisons():
    print("Generating Figure 2: First Two Weeks Time Series Comparison...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "target_squares_traffic.parquet"))

    # Two-week slice: Nov 01 to Nov 14, 2013
    df_two_weeks = df[(df["datetime"] >= "2013-11-01") & (df["datetime"] <= "2013-11-14 23:50:00")]
    target_squares = [5161, 5059, 5259, 4159, 4556]

    fig, axes = plt.subplots(len(target_squares), 1, figsize=(14, 2.5 * len(target_squares)), sharex=True)
    labels = {
        5161: "Square ID 5161 (Rank 1 - Commercial/Center)",
        5059: "Square ID 5059 (Rank 2 - Urban Core)",
        5259: "Square ID 5259 (Rank 3 - High Density)",
        4159: "Square ID 4159 (Bovisa / Mixed)",
        4556: "Square ID 4556 (San Siro / Event-Driven)"
    }

    for ax, sq_id in zip(axes, target_squares):
        sub = df_two_weeks[df_two_weeks["square_id"] == sq_id].sort_values("datetime")
        ax.plot(sub["datetime"], sub["internet"], linewidth=1.1, label=labels.get(sq_id, f"Square {sq_id}"))
        ax.set_ylabel("Internet Traffic", fontsize=9)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Date (Nov 1 - Nov 14, 2013)", fontsize=10)
    fig.suptitle("Internet Traffic Activity (First 2 Weeks Across 5 Key Areas)", fontsize=13, y=0.99)

    save_path = os.path.join(FIGURES_DIR, "fig2_two_weeks_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# 3. Conduct ACF/PACF and STL Decomposition on Rank 1 Area (5161)
def analyze_top_area():
    print("Performing In-Depth Diagnostics on Rank 1 Area (Square 5161)...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "target_squares_traffic.parquet"))
    top_area = 5161

    sub = df[df["square_id"] == top_area].drop_duplicates(subset=["datetime"]).set_index("datetime")
    series = sub["internet"].asfreq("10min").ffill()

    # Diagnostic A: ACF and PACF (144 steps = 24 hours)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(series, lags=288, ax=ax1, title="Autocorrelation Function (ACF) - Square ID 5161 (288 Lags = 48 Hours)")
    plot_pacf(series, lags=60, ax=ax2, method="ywm", title="Partial Autocorrelation Function (PACF) - Square ID 5161 (60 Lags = 10 Hours)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.grid(True, linestyle="--", alpha=0.4)

    acf_pacf_path = os.path.join(FIGURES_DIR, f"fig3_acf_pacf_square_{top_area}.png")
    plt.tight_layout()
    plt.savefig(acf_pacf_path, dpi=300)
    plt.close()
    print(f"Saved: {acf_pacf_path}")

    # Stationarity: Augmented Dickey-Fuller Test
    print("\n--- Augmented Dickey-Fuller (ADF) Test (Square 5161) ---")
    adf_res = adfuller(series.dropna())
    print(f"ADF Statistic:      {adf_res[0]:.4f}")
    print(f"p-value:            {adf_res[1]:.4e}")
    for k, v in adf_res[4].items():
        print(f"Critical Value ({k}): {v:.4f}")

    # Diagnostic B: Additive STL Decomposition (1 week slice = 1008 steps, period = 144 steps)
    decomp = seasonal_decompose(series.iloc[:1008], model="additive", period=144)
    fig = decomp.plot()
    fig.set_size_inches(12, 7)
    decomp_path = os.path.join(FIGURES_DIR, f"fig4_decomposition_square_{top_area}.png")
    plt.tight_layout()
    plt.savefig(decomp_path, dpi=300)
    plt.close()
    print(f"Saved: {decomp_path}")

if __name__ == "__main__":
    plot_spatial_distribution()
    plot_two_week_comparisons()
    analyze_top_area()