import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === DEBUG: Confirm working directory ===
print("📁 Current Working Directory:", os.getcwd())

# === Load CSV ===
try:
    df = pd.read_csv("features_windowed_s2_normalized.csv")
    print(f"✅ Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Failed to load CSV: {e}")
    exit()

# === Check if DataFrame is empty ===
if df.empty:
    print("❌ CSV is empty. Nothing to split.")
    exit()

# === Feature & Label Selection ===
X = df[["PERCLOS", "BlinkRate", "YawnRate"]]
y = df["KSS"]

# === Split: 70% train, 15% val, 15% test ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# === Save to CSV ===
try:
    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    X_val.to_csv("X_val.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    print("✅ Data split and saved successfully.")
except Exception as e:
    print(f"❌ Failed to save CSVs: {e}")
