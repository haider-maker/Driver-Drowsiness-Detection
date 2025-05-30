import os
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
feature_dir = "./features_output_s2"
video_prefix = "4-2"  # Target video subfolder
ear_vals = []
mar_vals = []

# === Load and filter files ===
files = sorted([f for f in os.listdir(feature_dir) if f.startswith(video_prefix) and f.endswith(".txt")])

print(f"📂 Found {len(files)} feature files for {video_prefix}")

for file in files:
    file_path = os.path.join(feature_dir, file)
    with open(file_path, 'r') as f:
        try:
            ear, mar = map(float, f.readline().strip().split())
            ear_vals.append(ear)
            mar_vals.append(mar)
        except:
            print(f"⚠️ Skipping invalid or empty file: {file}")

# === Plot EAR and MAR ===
plt.figure(figsize=(12, 5))
plt.plot(ear_vals, label='EAR (Eye Aspect Ratio)', color='blue')
plt.plot(mar_vals, label='MAR (Mouth Aspect Ratio)', color='orange')
plt.title(f"EAR and MAR over frames for video {video_prefix}")
plt.xlabel("Frame Index")
plt.ylabel("Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print Summary Stats ===
print(f"\n📊 EAR stats: mean={np.mean(ear_vals):.3f}, min={np.min(ear_vals):.3f}, max={np.max(ear_vals):.3f}")
print(f"📊 MAR stats: mean={np.mean(mar_vals):.3f}, min={np.min(mar_vals):.3f}, max={np.max(mar_vals):.3f}")
