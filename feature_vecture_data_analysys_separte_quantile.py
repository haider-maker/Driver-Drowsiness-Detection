import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# 1. Basic Statistics
print("Feature\t\tMin\tMax\tMean\tStd")
for key, col in FEATURE_VECTOR_MAP.items():
    vals = df[col].astype(float)
    print(f"{key:20s} min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f} median={vals.median():.4f} 25%={vals.quantile(0.25):.4f} 75%={vals.quantile(0.75):.4f} IQR={(vals.quantile(0.75)-vals.quantile(0.25)):.4f}")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(vals, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{key} Histogram")
    plt.xlabel(key)
    plt.ylabel("Frequency")
    plt.subplot(1,2,2)
    plt.boxplot(vals, vert=False)
    plt.title(f"{key} Boxplot")
    plt.xlabel(key)
    plt.tight_layout()
    plt.show()

# 4. Correlation Analysis

corr = df[[col for col in FEATURE_VECTOR_MAP.values()]].astype(float).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()