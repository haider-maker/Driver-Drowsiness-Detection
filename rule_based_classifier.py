import pandas as pd

# === CONFIG ===
csv_path = "features_windowed_s2.csv"  # Update this if your file has a different name
output_csv = "rule_based_predictions.csv"

# === Load Data ===
df = pd.read_csv(csv_path)

# === Define Rule-Based Fatigue Classifier ===
def rule_based_kss_class(perclos, blink_rate, yawn_rate):
    if perclos >= 0.035 or blink_rate >= 30 or yawn_rate >= 5:
        return 2  # High
    elif perclos >= 0.015 or blink_rate >= 12 or yawn_rate >= 1:
        return 1  # Moderate
    else:
        return 0  # Low

df["PredictedClass"] = df.apply(lambda row: rule_based_kss_class(row["PERCLOS"], row["BlinkRate"], row["YawnRate"]), axis=1)

# === Map Actual KSS to Classes for Evaluation ===
def kss_to_class(kss):
    if kss <= 3:
        return 0
    elif kss <= 6:
        return 1
    else:
        return 2

df["ActualClass"] = df["KSS"].apply(kss_to_class)

# === Save Result ===
df.to_csv(output_csv, index=False)
print(f"✅ Saved rule-based classification results to: {output_csv}")
