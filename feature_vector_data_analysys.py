"""
Feature Vector Data Analysis (Percentile Thresholds)
---------------------------------------------------
This script loads driver feature data, calculates threshold suggestions for each feature
using user-defined percentiles, and visualizes distributions with threshold lines.
It prints a ready-to-use Python dictionary for thresholds.

Steps performed:
    1. Load feature data from CSV.
    2. For each feature:
        - Calculate min, low, high, max values based on percentiles.
        - Print threshold ranges for Low, Moderate, High.
        - Plot histogram and boxplot with threshold lines.
    3. Print a Python dictionary for all thresholds.

User adjustable parameters:
    LOW_PERCENTILE: Lower percentile for threshold (e.g., 0.30 for 30th percentile).
    HIGH_PERCENTILE: Higher percentile for threshold (e.g., 0.70 for 70th percentile).

Dependencies:
    pandas, matplotlib, seaborn

Usage:
    python feature_vector_data_analysys.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- User adjustable percentiles for threshold sensitivity ---
LOW_PERCENTILE = 0.30   # e.g., 0.20 for 20th percentile
HIGH_PERCENTILE = 0.70  # e.g., 0.80 for 80th percentile

# Load the data
df = pd.read_csv("Feature_vector.csv")

# Map your feature names to the CSV columns
FEATURE_VECTOR_MAP = {
    "PERCLOS": "PERCLOS",
    "BlinkRate": "BlinkRate",
    "YawningRate": "YawnRate",
    "SteeringEntropy": "Steering Entropy",
    "SteeringReversalRate": "SRR",
    "SteeringStd": "SAV",
    "OffsetStd": "SDLP",
    "LaneDepartureFrequency": "Lane Departure Frequency",
    "LaneKeepingRatio": "Lane Keeping Ratio"
}

print(f"\nThreshold suggestions using {int(LOW_PERCENTILE*100)}%/{int(HIGH_PERCENTILE*100)}% percentiles:\n")
threshold_dict = {}

for key, col in FEATURE_VECTOR_MAP.items():
    vals = df[col].astype(float)
    min_v = vals.min()
    low_v = vals.quantile(LOW_PERCENTILE)
    high_v = vals.quantile(HIGH_PERCENTILE)
    max_v = vals.max()
    print(f"{key}:")
    print(f"  Low:      {min_v:.4f} – {low_v:.4f}")
    print(f"  Moderate: {low_v:.4f} – {high_v:.4f}")
    print(f"  High:     {high_v:.4f} – {max_v:.4f}\n")
    threshold_dict[key] = {
        "Low": (float(f"{min_v:.4f}"), float(f"{low_v:.4f}")),
        "Moderate": (float(f"{low_v:.4f}"), float(f"{high_v:.4f}")),
        "High": (float(f"{high_v:.4f}"), float(f"{max_v:.4f}"))
    }
    # Visualization: histogram and boxplot with threshold lines
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(vals, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{key} Histogram")
    plt.xlabel(key)
    plt.ylabel("Frequency")
    plt.axvline(low_v, color='orange', linestyle='--', label=f'Low ({LOW_PERCENTILE*100:.0f}%)')
    plt.axvline(high_v, color='red', linestyle='--', label=f'High ({HIGH_PERCENTILE*100:.0f}%)')
    plt.legend()
    plt.subplot(1,2,2)
    plt.boxplot(vals, vert=False)
    plt.title(f"{key} Boxplot")
    plt.xlabel(key)
    plt.tight_layout()
    plt.show()

# Print out a ready-to-use Python threshold dictionary
print("\nSuggested FEATURE_THRESHOLDS dictionary:\n")
print("FEATURE_THRESHOLDS = {")
for key, v in threshold_dict.items():
    print(f'    "{key}": {v},')
print("}")