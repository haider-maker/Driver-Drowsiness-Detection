import pandas as pd

# === CONFIG ===
csv_path = "features_windowed_s2.csv"  # Change if your CSV file name is different

# === Load CSV ===
df = pd.read_csv(csv_path)

# === Convert YawnRate to float (if it's read as string)
df["YawnRate"] = pd.to_numeric(df["YawnRate"], errors="coerce")

# === Filter rows where YawnRate > 0
yawn_df = df[df["YawnRate"] > 0]

# === Print and optionally save
print(f"🟠 Found {len(yawn_df)} frames with YawnRate > 0:\n")
print(yawn_df[["Video", "Frame", "YawnRate", "Timestamp"]])

# === Optional: Save to CSV
yawn_df.to_csv("yawn_detected_frames.csv", index=False)
print("\n✅ Saved filtered frames to 'yawn_detected_frames.csv'")
