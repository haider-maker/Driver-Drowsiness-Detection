import os
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
base_dir = "data/EAR_MAR_dlib_output"
video_prefix = "14-3"  # Target subfolder
target_dir = os.path.join(base_dir, video_prefix)

ear_vals = []
mar_vals = []

# === Sanity check ===
if not os.path.isdir(target_dir):
    print(f"❌ Subfolder not found: {target_dir}")
    exit()

# === Load and filter files ===
files = sorted([f for f in os.listdir(target_dir) if f.endswith(".txt")])
print(f"📂 Found {len(files)} feature files in '{video_prefix}'")

for file in files:
    file_path = os.path.join(target_dir, file)
    with open(file_path, 'r') as f:
        try:
            ear, mar = map(float, f.readline().strip().split())
            ear_vals.append(ear)
            mar_vals.append(mar)
        except:
            print(f"⚠️ Skipping invalid or empty file: {file}")

# === Check data availability ===
if not ear_vals or not mar_vals:
    print("❌ No valid EAR or MAR values found.")
    exit()

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
