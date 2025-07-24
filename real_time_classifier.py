import pandas as pd
import time
from datetime import datetime
import os

# Mapping between feature names in source and internal keys
FEATURE_VECTOR_MAP = {
    "PERCLOS": "PERCLOS",
    "BlinkRate": "BlinkRate",
    "YawningRate": "YawnRate",
    "SteeringEntropy": "Steering Entropy",
    "SteeringReversalRate": "SRR",
    "SteeringStd": "SAV",
    "OffsetStd": "SDLP",
    "LaneDepartureFrequency": "Lane Departure Frequency",
    "LaneKeepingRatio": "Lane Keeping Ratio",
    "Timestamp": "Timestamp"
}

def remap_feature_vector_row(row):
    """
    Remap a row from the source feature vector to the internal feature names.

    Args:
        row (pd.Series): Row from the source DataFrame.

    Returns:
        dict: Remapped row with internal feature names as keys.
    """
    mapped = {}
    for key, src in FEATURE_VECTOR_MAP.items():
        mapped[key] = row[src]
    return mapped

# Thresholds for classifying each feature into Low, Moderate, or High fatigue levels
FEATURE_THRESHOLDS = {
    "PERCLOS": {'Low': (0.005, 0.055), 'Moderate': (0.055, 0.1092), 'High': (0.1092, 0.2283)},
    "BlinkRate": {'Low': (0.0833, 0.4833), 'Moderate': (0.4833, 0.8), 'High': (0.8, 1.2667)},
    "YawningRate": {'Low': (0.0, 0.0), 'Moderate': (0.0, 0.0167), 'High': (0.0167, 0.0667)},
    "SteeringEntropy": {'Low': (2.0036, 2.5794), 'Moderate': (2.5794, 2.8498), 'High': (2.8498, 2.9994)},
    "SteeringReversalRate": {'Low': (0.1833, 0.45), 'Moderate': (0.45, 0.75), 'High': (0.75, 1.15)},
    "SteeringStd": {'Low': (0.021, 0.0347), 'Moderate': (0.0347, 0.0391), 'High': (0.0391, 0.1198)},
    "OffsetStd": {'Low': (0.2488, 0.373), 'Moderate': (0.373, 0.5668), 'High': (0.5668, 0.9563)},
    "LaneDepartureFrequency": {'Low': (0.0, 0.0), 'Moderate': (0.0, 0.2333), 'High': (0.2333, 1.0167)},
    "LaneKeepingRatio": {'Low': (0.898, 0.9767), 'Moderate': (0.9767, 1.0), 'High': (1.0, 1.0)},
}

def classify_feature(value, feature_name):
    """
    Classify a feature value into a fatigue level based on predefined thresholds.

    Args:
        value (float): Feature value.
        feature_name (str): Name of the feature.

    Returns:
        str: Fatigue level ('Low', 'Moderate', 'High', or 'Unknown').
    """
    for level, (low, high) in FEATURE_THRESHOLDS[feature_name].items():
        if low <= value <= high:
            return level
    return "Unknown"

def majority_classification(labels):
    """
    Determine the majority fatigue level from a list of labels.

    Args:
        labels (list): List of fatigue levels.

    Returns:
        str: Majority fatigue level.
    """
    counts = {level: labels.count(level) for level in ["High", "Moderate", "Low"]}
    if counts["High"] >= 2:
        return "High"
    elif counts["Moderate"] >= 2:
        return "Moderate"
    else:
        return "Low"

def classify_row(row):
    """
    Classify a row into camera, steering, and lane fatigue levels.

    Args:
        row (dict or pd.Series): Feature vector row.

    Returns:
        tuple: (camera_level, steering_level, lane_level)
    """
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

PROCESSED_LOG = set()  # Stores processed timestamps to avoid duplicates
CLASSIFIED_FILE = "real_captured_fatigue_classified_30_70.csv"
SOURCE_FILE = "Feature_vector.csv"  # Source file for real-time data

# Output columns for the classified CSV file
output_columns = [
    "ID", "timestamp", "Blink Rate", "Yawning Rate", "PERCLOS", "SDLP",
    "Lane Keeping Ratio", "Lane Departure Frequency", "Steering Entropy",
    "SRR", "SAV", "fatigue_camera_level", "fatigue_steering_level", "fatigue_lane_level",
    "fan", "music", "vibration", "reason"
]

# Create output file with header if it does not exist or is empty
if not os.path.exists(CLASSIFIED_FILE) or os.stat(CLASSIFIED_FILE).st_size == 0:
    pd.DataFrame(columns=output_columns).to_csv(CLASSIFIED_FILE, index=False)

# Load already processed timestamps to avoid duplicate processing
try:
    df_classified = pd.read_csv(CLASSIFIED_FILE)
    if "timestamp" in df_classified.columns:
        PROCESSED_LOG = set(df_classified["timestamp"].astype(str))
except Exception as e:
    print(f"Warning: Could not load processed log: {e}")

print("🚀 Monitoring started. Watching for new data...")

try:
    while True:
        # Read source feature vector file
        df_raw = pd.read_csv(SOURCE_FILE)
        # Remap columns if using Feature_vector.csv
        if "Feature_vector.csv" in SOURCE_FILE:
            df = pd.DataFrame([remap_feature_vector_row(row) for _, row in df_raw.iterrows()])
        else:
            df = df_raw

        # Filter new rows that have not been processed yet
        new_rows = df[~df["Timestamp"].isin(PROCESSED_LOG)]

        if not new_rows.empty:
            results = []
            for _, row in new_rows.iterrows():
                cf, sf, lf = classify_row(row)
                # Build output row for classified CSV
                out_row = {
                    "timestamp": row.get("Timestamp", ""),
                    "Blink Rate": row.get("BlinkRate", ""),
                    "Yawning Rate": row.get("YawningRate", row.get("YawnRate", "")),
                    "PERCLOS": row.get("PERCLOS", ""),
                    "SDLP": row.get("OffsetStd", row.get("SDLP", "")),
                    "Lane Keeping Ratio": row.get("LaneKeepingRatio", ""),
                    "Lane Departure Frequency": row.get("LaneDepartureFrequency", ""),
                    "Steering Entropy": row.get("SteeringEntropy", ""),
                    "SRR": row.get("SteeringReversalRate", row.get("SRR", "")),
                    "SAV": row.get("SteeringStd", row.get("SAV", "")),
                    "fatigue_camera_level": cf,
                    "fatigue_steering_level": sf,
                    "fatigue_lane_level": lf,
                    "fan": "",
                    "music": "",
                    "vibration": "",
                    "reason": ""
                }
                results.append(out_row)
                PROCESSED_LOG.add(row["Timestamp"])

            # Assign unique ID and reorder columns
            try:
                df_existing = pd.read_csv(CLASSIFIED_FILE)
                max_id = df_existing["ID"].max() if not df_existing.empty else 0
            except Exception:
                max_id = 0

            df_out = pd.DataFrame(results)
            df_out["ID"] = range(max_id + 1, max_id + 1 + len(df_out))
            df_out = df_out[output_columns]
            # Append new rows to classified file
            write_header = not os.path.exists(CLASSIFIED_FILE) or os.stat(CLASSIFIED_FILE).st_size == 0
            df_out.to_csv(CLASSIFIED_FILE, mode="a", index=False, header=False)
            print(f"✅ Processed {len(df_out)} new rows at {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(2)  # Wait before checking for new data again

except KeyboardInterrupt:
    print("🛑 Real-time monitor stopped.")

