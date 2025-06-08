import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# === CONFIGURATION ===
dlib_landmark_dir = Path("data/landmarks_dlib")
s2_landmark_base = Path("data/frames_mapped_s2")
frame_base = Path("data/frames_mapped_s2")
output_csv = "landmark_comparison.csv"
num_visual_samples = 5  # number of images to visualize

# === Helper: Load landmarks ===
def load_s2_landmarks(path):
    try:
        numbers = list(map(float, path.read_text().strip().split()))
        return np.array(list(zip(numbers[::2], numbers[1::2]))) if len(numbers) == 136 else None
    except:
        return None

def load_dlib_landmarks(path):
    try:
        lines = path.read_text().strip().splitlines()
        return np.array([tuple(map(int, line.strip().split())) for line in lines]) if len(lines) == 68 else None
    except:
        return None

# === Comparison Loop ===
results = []
distances_all = []
valid_samples = []

for dlib_file in sorted(dlib_landmark_dir.glob("*.txt")):
    name = dlib_file.stem  # e.g., 11-1_frame_0756
    if "_frame_" not in name:
        continue

    video_id, frame_id = name.split("_frame_")
    s2_file = s2_landmark_base / video_id / f"frame_{frame_id}.txt"
    img_path = frame_base / video_id / f"frame_{frame_id}.jpg"

    s2_lm = load_s2_landmarks(s2_file)
    dlib_lm = load_dlib_landmarks(dlib_file)

    if s2_lm is None or dlib_lm is None or s2_lm.shape != (68, 2) or dlib_lm.shape != (68, 2):
        continue

    dist = np.linalg.norm(s2_lm - dlib_lm, axis=1)
    avg_dist = np.mean(dist)
    results.append((name, avg_dist))
    distances_all.append(avg_dist)
    valid_samples.append((img_path, s2_lm, dlib_lm, name))

# === Save CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Avg_Euclidean_Distance"])
    writer.writerows(results)

# === Summary ===
print("\n📊 === SUMMARY ===")
print(f"✅ Total Compared: {len(results)}")
print(f"📉 Mean Distance: {np.mean(distances_all):.2f}")
print(f"📈 Max Distance : {np.max(distances_all):.2f}")
print(f"📉 Min Distance : {np.min(distances_all):.2f}")
print(f"💾 CSV Saved To : {output_csv}")

# === VISUALIZE RANDOM SAMPLES ===
print(f"\n🖼️ Visualizing {num_visual_samples} random samples...")
samples = random.sample(valid_samples, min(num_visual_samples, len(valid_samples)))

for img_path, s2_lm, dlib_lm, frame_name in samples:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to load: {img_path}")
        continue

    for (x, y) in s2_lm:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)  # Green for S2

    for (x, y) in dlib_lm:
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)  # Red for Dlib

    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{frame_name} (Green=S2, Red=Dlib)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === HISTOGRAM of DISTANCES ===
plt.figure(figsize=(8, 4))
plt.hist(distances_all, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Avg. Euclidean Distances (Dlib vs S2)")
plt.xlabel("Distance")
plt.ylabel("Frame Count")
plt.grid(True)
plt.tight_layout()
plt.show()
