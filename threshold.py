import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === CONFIG ===
feature_dir = "./features_output_s2"
video_ear = defaultdict(list)
video_mar = defaultdict(list)

print("🔍 Starting feature file loading...")

# === Load all features grouped by video prefix ===
total_files = 0
valid_files = 0
skipped_files = 0
video_count = 0

for idx, file in enumerate(sorted(os.listdir(feature_dir)), 1):
    if not file.endswith(".txt"):
        continue

    total_files += 1

    try:
        video_prefix = file.split("_")[0]
        file_path = os.path.join(feature_dir, file)
        with open(file_path, 'r') as f:
            ear, mar = map(float, f.readline().strip().split())
            video_ear[video_prefix].append(ear)
            video_mar[video_prefix].append(mar)
            valid_files += 1
    except Exception as e:
        print(f"⚠️ Skipping invalid file: {file}")
        skipped_files += 1
        continue

    if idx % 500 == 0:
        print(f"📈 Processed {idx} files...")

print(f"\n📂 Total files scanned: {total_files}")
print(f"✅ Valid files processed: {valid_files}")
print(f"⚠️ Skipped files: {skipped_files}")
print(f"🎥 Unique videos found: {len(video_ear)}\n")

# === Plot and Stats per Video ===
all_ear = []
all_mar = []

for count, vid in enumerate(sorted(video_ear.keys()), 1):
    ears = video_ear[vid]
    mars = video_mar[vid]

    all_ear.extend(ears)
    all_mar.extend(mars)

    print(f"\n📊 Generating plot for video {vid} ({count}/{len(video_ear)})...")

    plt.figure(figsize=(12, 4))
    plt.plot(ears, label='EAR', color='blue')
    plt.plot(mars, label='MAR', color='orange')
    plt.title(f"[{vid}] EAR & MAR over frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n📹 Video: {vid}")
    print(f"  📊 EAR → mean: {np.mean(ears):.3f}, min: {np.min(ears):.3f}, max: {np.max(ears):.3f}")
    print(f"  📊 MAR → mean: {np.mean(mars):.3f}, min: {np.min(mars):.3f}, max: {np.max(mars):.3f}")

# === Global Summary ===
print("\n📡 Finished all videos.")
print("🧠 GLOBAL STATS ACROSS ALL VIDEOS:")
print(f"📊 EAR → mean: {np.mean(all_ear):.3f}, min: {np.min(all_ear):.3f}, max: {np.max(all_ear):.3f}")
print(f"📊 MAR → mean: {np.mean(all_mar):.3f}, min: {np.min(all_mar):.3f}, max: {np.max(all_mar):.3f}")
