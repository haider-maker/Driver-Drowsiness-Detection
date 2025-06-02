import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# === CONFIG ===
csv_path = "features_windowed_improved.csv"
target_video = "6-2"

# === Load CSV ===
df = pd.read_csv(csv_path)
df = df[df["Video"] == target_video].reset_index(drop=True)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])  
start_time = df["Timestamp"].min()
df["RelativeTimeSec"] = (df["Timestamp"] - start_time).dt.total_seconds()

# === Plotting ===
plt.figure(figsize=(14, 6))
plt.plot(df["RelativeTimeSec"], df["YawnRate"], label="Yawn Rate", color='orange', marker='o')
plt.plot(df["RelativeTimeSec"], df["PERCLOS"], label="PERCLOS", linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.title(f"Yawn Rate and PERCLOS over time - {target_video}")
plt.ylabel("Rate")
plt.xlabel("Time (seconds from start)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
