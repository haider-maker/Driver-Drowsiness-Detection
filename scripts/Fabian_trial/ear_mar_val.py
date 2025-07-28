import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIG ===
subject = "Ghulam"
video_id = "KSS_4_Vid_1"

base_dir = f"data/Subjects_EAR_MAR_output/{subject}/{video_id}"

ear_vals = []
mar_vals = []
timestamps_float = []

# === Sanity check ===
if not os.path.isdir(base_dir):
    print(f"❌ Folder not found: {base_dir}")
    exit()

# === Load all .txt files ===
files = sorted([f for f in os.listdir(base_dir) if f.endswith(".txt")])
print(f"📂 Found {len(files)} feature files in folder: {base_dir}")

for file in files:
    file_path = os.path.join(base_dir, file)

    # Extract timestamp float from filename
    try:
        stem_str = Path(file).stem.strip()
        timestamp_float = float(stem_str)
    except Exception as e:
        print(f"⚠️ Skipping file with unexpected name format: {file} ({e})")
        continue

    with open(file_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            print(f"⚠️ Skipping empty file: {file}")
            continue
        try:
            ear, mar = map(float, line.split())
            ear_vals.append(ear)
            mar_vals.append(mar)
            timestamps_float.append(timestamp_float)
        except:
            print(f"⚠️ Skipping invalid data in file: {file}")

# === Check data availability ===
if not ear_vals or not mar_vals:
    print("❌ No valid EAR or MAR values found.")
    exit()

# === Sort all lists by timestamp ===
sorted_indices = np.argsort(timestamps_float)
timestamps_sorted = np.array(timestamps_float)[sorted_indices]
ear_vals_sorted = np.array(ear_vals)[sorted_indices]
mar_vals_sorted = np.array(mar_vals)[sorted_indices]

# === Plot EAR and MAR ===
plt.figure(figsize=(12, 5))
plt.plot(timestamps_sorted, ear_vals_sorted, label='EAR (Eye Aspect Ratio)', color='blue')
plt.plot(timestamps_sorted, mar_vals_sorted, label='MAR (Mouth Aspect Ratio)', color='orange')
plt.title(f"EAR and MAR over time for {subject} - {video_id}")
plt.xlabel("Timestamp (float seconds since epoch)")
plt.ylabel("Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print Summary Stats ===
print(f"\n📊 EAR stats: mean={np.mean(ear_vals_sorted):.3f}, min={np.min(ear_vals_sorted):.3f}, max={np.max(ear_vals_sorted):.3f}")
print(f"📊 MAR stats: mean={np.mean(mar_vals_sorted):.3f}, min={np.min(mar_vals_sorted):.3f}, max={np.max(mar_vals_sorted):.3f}")
