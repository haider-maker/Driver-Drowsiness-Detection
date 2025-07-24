import pandas as pd
import random
import time
from datetime import datetime

# === Realistic feature value ranges ===
RANGES = {
    "PERCLOS": (5, 50),  # Percentage of eye closure
    "BlinkRate": (10, 30),  # Blinks per minute
    "YawningRate": (0, 3),  # Yawns per minute
    "SteeringEntropy": (0.5, 5),  # Steering entropy value
    "SteeringReversalRate": (5, 35),  # Steering reversal rate
    "SteeringStd": (0.015, 0.05),  # Standard deviation of steering
    "OffsetStd": (10, 60),  # Standard deviation of lane offset
    "LaneDepartureFrequency": (1, 5),  # Lane departures per minute
    "LaneKeepingRatio": (0.6, 1)  # Ratio of time spent keeping lane
}

# Desired column order: Timestamp first
FEATURE_COLUMNS = ["Timestamp"] + list(RANGES.keys())

def generate_sample():
    """
    Generate a single row of simulated driver feature data.

    Returns:
        dict: Dictionary containing timestamp and random feature values.
    """
    row = {}
    row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for feature, (low, high) in RANGES.items():
        row[feature] = round(random.uniform(low, high), 4)
    return row

csv_path = "real_captured_features.csv"

# Create file if doesn't exist, with correct column order
try:
    pd.read_csv(csv_path)
except FileNotFoundError:
    print("📁 Creating new real_captured_features.csv")
    pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(csv_path, index=False)

print("🚀 Starting real-time input simulation...")

try:
    while True:
        new_row = generate_sample()
        df_new = pd.DataFrame([new_row], columns=FEATURE_COLUMNS)
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
        print(f"🟢 Appended row at {new_row['Timestamp']}")
        time.sleep(5)
except KeyboardInterrupt:
    print("🛑 Real-time simulation stopped.")
