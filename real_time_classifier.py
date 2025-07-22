import pandas as pd
import time
from datetime import datetime


FEATURE_VECTOR_MAP = {
    "PERCLOS": "PERCLOS",
    "BlinkRate": "BlinkRate",  # You may need to scale or convert
    "YawningRate": "YawnRate", # Rename YawnRate to YawningRate
    "SteeringEntropy": "Steering Entropy",
    "SteeringReversalRate": "SRR",
    "SteeringStd": "SAV",
    "OffsetStd": "SDLP",
    "LaneDepartureFrequency": "Lane Departure Frequency",
    "LaneKeepingRatio": "Lane Keeping Ratio",
    "Timestamp": "Timestamp"
}

def remap_feature_vector_row(row):
    # Map and rename columns, and convert to expected types if needed
    mapped = {}
    for key, src in FEATURE_VECTOR_MAP.items():
        mapped[key] = row[src]
    return mapped

# === Thresholds for classification ===
## Threshold suggestions using 25%/75% percentiles:
# FEATURE_THRESHOLDS = {
#     "PERCLOS": {'Low': (0.005, 0.0508), 'Moderate': (0.0508, 0.1525), 'High': (0.1525, 0.2283)},
#     "BlinkRate": {'Low': (0.0833, 0.4667), 'Moderate': (0.4667, 0.8667), 'High': (0.8667, 1.2667)},
#     "YawningRate": {'Low': (0.0, 0.0), 'Moderate': (0.0, 0.0167), 'High': (0.0167, 0.0667)},
#     "SteeringEntropy": {'Low': (2.0036, 2.5481), 'Moderate': (2.5481, 2.8723), 'High': (2.8723, 2.9994)},
#     "SteeringReversalRate": {'Low': (0.1833, 0.4333), 'Moderate': (0.4333, 0.75), 'High': (0.75, 1.15)},
#     "SteeringStd": {'Low': (0.021, 0.0335), 'Moderate': (0.0335, 0.0396), 'High': (0.0396, 0.1198)},
#     "OffsetStd": {'Low': (0.2488, 0.3568), 'Moderate': (0.3568, 0.6038), 'High': (0.6038, 0.9563)},
#     "LaneDepartureFrequency": {'Low': (0.0, 0.0), 'Moderate': (0.0, 0.25), 'High': (0.25, 1.0167)},
#     "LaneKeepingRatio": {'Low': (0.898, 0.975), 'Moderate': (0.975, 1.0), 'High': (1.0, 1.0)},
# }
## Threshold suggestions using 30%/70% percentiles: 
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
CLASSIFIED_FILE = "real_captured_fatigue_classified_30_70.csv"
SOURCE_FILE = "Feature_vector.csv"  # Change to "real_captured_features.csv" if needed

# Create output file with header if not exists
try:
    pd.read_csv(CLASSIFIED_FILE)
except FileNotFoundError:
    pd.DataFrame(columns=[
                "ID", "timestamp", "Blink Rate", "Yawning Rate", "PERCLOS", "SDLP",
                "Steering Entropy", "Lane Keeping Ratio", "Lane Departure Frequency",
                "SRR", "SAV", "CF", "SF", "LF", "fan", "music", "vibration", "reason"
    ]).to_csv(CLASSIFIED_FILE, index=False)

# --- Load already processed timestamps to avoid duplicates ---
try:
    df_classified = pd.read_csv(CLASSIFIED_FILE)
    if "Timestamp" in df_classified.columns:
        PROCESSED_LOG = set(df_classified["Timestamp"].astype(str))
except Exception as e:
    print(f"Warning: Could not load processed log: {e}")


print("🚀 Monitoring started. Watching for new data...")

output_columns = [
                "ID", "timestamp", "Blink Rate", "Yawning Rate", "PERCLOS", "SDLP",
                "Steering Entropy", "Lane Keeping Ratio", "Lane Departure Frequency",
                "SRR", "SAV", "CF", "SF", "LF", "fan", "music", "vibration", "reason"
            ]

COLUMN_MAP = {
                "Timestamp": "timestamp",
                "BlinkRate": "Blink Rate",
                "YawningRate": "Yawning Rate",
                "PERCLOS": "PERCLOS",
                "OffsetStd": "SDLP",
                "SteeringEntropy": "Steering Entropy",
                "LaneKeepingRatio": "Lane Keeping Ratio",
                "LaneDepartureFrequency": "Lane Departure Frequency",
                "SteeringReversalRate": "SRR",
                "SteeringStd": "SAV",
                "CF": "CF",
                "SF": "SF",
                "LF": "LF"
            }

try:
    while True:
        df_raw = pd.read_csv(SOURCE_FILE)
        # Remap columns if using Feature_vector.csv
        if "Feature_vector.csv" in SOURCE_FILE:
            df = pd.DataFrame([remap_feature_vector_row(row) for _, row in df_raw.iterrows()])
        else:
            df = df_raw

        new_rows = df[~df["Timestamp"].isin(PROCESSED_LOG)]

        if not new_rows.empty:
            results = []
            for _, row in new_rows.iterrows():
                cf, sf, lf = classify_row(row)
                mapped_row = {COLUMN_MAP.get(k, k): v for k, v in row.to_dict().items() if k in COLUMN_MAP}
                mapped_row["CF"] = cf
                mapped_row["SF"] = sf
                mapped_row["LF"] = lf
                # Fill extra columns with default values
                mapped_row["fan"] = ""
                mapped_row["music"] = ""
                mapped_row["vibration"] = ""
                mapped_row["reason"] = ""
                results.append(mapped_row)
            # results = []
            # for _, row in new_rows.iterrows():
            #     cf, sf, lf = classify_row(row)
            #     results.append({
            #         **row.to_dict(),
            #         "CF": cf,
            #         "SF": sf,
            #         "LF": lf
            #     })
                PROCESSED_LOG.add(row["Timestamp"])  # Avoid reprocessing

            try:
                df_existing = pd.read_csv(CLASSIFIED_FILE)
                max_id = df_existing["ID"].max() if not df_existing.empty else 0
            except Exception:
                max_id = 0
            # Specify the correct column order here:
            
            df_out = pd.DataFrame(results, columns=output_columns)
            df_out["ID"] = range(1, len(df_out) + 1)
            df_out.to_csv(CLASSIFIED_FILE, mode="a", index=False, header=False)
            print(f"✅ Processed {len(df_out)} new rows at {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(5)  # Check every 5 seconds

except KeyboardInterrupt:
    print("🛑 Real-time monitor stopped.")


