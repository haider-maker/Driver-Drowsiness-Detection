import os
import numpy as np
from scipy.signal import medfilt
from pathlib import Path

# === CONFIG ===
input_dir = Path("./features_output")            # Raw EAR & MAR files
output_dir = Path("./features_filtered")         # Output for filtered values
os.makedirs(output_dir, exist_ok=True)

# === Parameters ===
window_size = 5  # must be odd, adjust as needed

# === Collect & Group by video ===
video_dict = {}

for file in sorted(input_dir.glob("*.txt")):
    parts = file.stem.split("_")
    if len(parts) < 2:
        continue
    video = parts[0]
    if video not in video_dict:
        video_dict[video] = []
    video_dict[video].append(file)

# === Process each video ===
for video, files in video_dict.items():
    files = sorted(files, key=lambda f: int(f.stem.split("_frame_")[1].split("_")[0]))
    ears, mars, paths = [], [], []

    for path in files:
        with open(path) as f:
            try:
                ear, mar = map(float, f.read().strip().split())
                ears.append(ear)
                mars.append(mar)
                paths.append(path.name)
            except:
                continue

    if len(ears) < window_size:
        print(f"⚠️ Not enough frames in {video} to filter.")
        continue

    # Apply median filter
    filtered_ears = medfilt(ears, kernel_size=window_size)
    filtered_mars = medfilt(mars, kernel_size=window_size)

    # Save to new files
    for i, fname in enumerate(paths):
        out_path = output_dir / fname
        with open(out_path, "w") as f:
            f.write(f"{filtered_ears[i]:.6f} {filtered_mars[i]:.6f}\n")

    print(f"✅ Filtered and saved EAR/MAR for {video}: {len(ears)} frames.")

print("🎉 All videos filtered successfully.")
