import pandas as pd
import time
from datetime import datetime

# === Thresholds for classification ===
FEATURE_THRESHOLDS = {
    "PERCLOS": {"Low": (0, 15), "Moderate": (15, 30), "High": (30, 100)},
    "BlinkRate": {"Low": (10, 15), "Moderate": (15, 18), "High": (18, 50)},
    "YawningRate": {"Low": (0, 0), "Moderate": (1, 1), "High": (2, 5)},
    "SteeringEntropy": {"Low": (0, 1), "Moderate": (1, 3), "High": (3, 10)},
    "SteeringReversalRate": {"Low": (25, 35), "Moderate": (15, 25), "High": (0, 15)},
    "SteeringStd": {"Low": (0, 0.025), "Moderate": (0.025, 0.035), "High": (0.035, 0.1)},
    "OffsetStd": {"Low": (0, 20), "Moderate": (20, 35), "High": (35, 100)},
    "LaneDepartureFrequency": {"Low": (0, 2.5), "Moderate": (2.5, 3.5), "High": (3.5, 10)},
    "LaneKeepingRatio": {"Low": (0.85, 1), "Moderate": (0.7, 0.85), "High": (0, 0.7)}
}

def classify_feature(value, feature_name):
    for level, (low, high) in FEATURE_THRESHOLDS[feature_name].items():
        if low <= value <= high:
            return level
    return "Unknown"

def majority_classification(labels):
    counts = {level: labels.count(level) for level in ["High", "Moderate", "Low"]}
    if counts["High"] >= 2:
        return "High"
    elif counts["Moderate"] >= 2:
        return "Moderate"
    else:
        return "Low"

def classify_row(row):
    cam_labels = [
        classify_feature(row["PERCLOS"], "PERCLOS"),
        classify_feature(row["BlinkRate"], "BlinkRate"),
        classify_feature(row["YawningRate"], "YawningRate"),
    ]
    steer_labels = [
        classify_feature(row["SteeringEntropy"], "SteeringEntropy"),
        classify_feature(row["SteeringReversalRate"], "SteeringReversalRate"),
        classify_feature(row["SteeringStd"], "SteeringStd"),
    ]
    lane_labels = [
        classify_feature(row["OffsetStd"], "OffsetStd"),
        classify_feature(row["LaneDepartureFrequency"], "LaneDepartureFrequency"),
        classify_feature(row["LaneKeepingRatio"], "LaneKeepingRatio"),
    ]
    return majority_classification(cam_labels), majority_classification(steer_labels), majority_classification(lane_labels)

# === Real-time processing loop ===
PROCESSED_LOG = set()
CLASSIFIED_FILE = "real_captured_fatigue_classified.csv"
SOURCE_FILE = "real_captured_features.csv"

# Create output file with header if not exists
try:
    pd.read_csv(CLASSIFIED_FILE)
except FileNotFoundError:
    pd.DataFrame(columns=[
        "Timestamp", "PERCLOS", "BlinkRate", "YawningRate",
        "SteeringEntropy", "SteeringReversalRate", "SteeringStd",
        "OffsetStd", "LaneDepartureFrequency", "LaneKeepingRatio",
        "CF", "SF", "LF"
    ]).to_csv(CLASSIFIED_FILE, index=False)

# --- Load already processed timestamps to avoid duplicates ---
try:
    df_classified = pd.read_csv(CLASSIFIED_FILE)
    if "Timestamp" in df_classified.columns:
        PROCESSED_LOG = set(df_classified["Timestamp"].astype(str))
except Exception as e:
    print(f"Warning: Could not load processed log: {e}")


print("🚀 Monitoring started. Watching for new data...")

try:
    while True:
        df = pd.read_csv(SOURCE_FILE)

        new_rows = df[~df["Timestamp"].isin(PROCESSED_LOG)]

        if not new_rows.empty:
            results = []
            for _, row in new_rows.iterrows():
                cf, sf, lf = classify_row(row)
                results.append({
                    **row.to_dict(),
                    "CF": cf,
                    "SF": sf,
                    "LF": lf
                })
                PROCESSED_LOG.add(row["Timestamp"])  # Avoid reprocessing

            df_out = pd.DataFrame(results)
            df_out.to_csv(CLASSIFIED_FILE, mode="a", index=False, header=False)
            print(f"✅ Processed {len(df_out)} new rows at {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(5)  # Check every 5 seconds

except KeyboardInterrupt:
    print("🛑 Real-time monitor stopped.")
