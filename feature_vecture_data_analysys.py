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
    # Visualization
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