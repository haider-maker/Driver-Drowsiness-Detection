import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import cv2
import csv

# === CONFIGURATION ===
dlib_landmark_dir = Path("data/landmarks_dlib")
s2_landmark_base = Path("data/frames_mapped_s2")
image_base = Path("data/frames_mapped_s2")
output_csv = "landmark_eye_mouth_comparison.csv"

# Eye and Mouth Landmark Indices (68-point)
EYE_MOUTH_IDX = list(range(36, 48)) + list(range(48, 68))  # Eyes (36–47), Mouth (48–67)

def load_s2_landmarks(path):
    try:
        numbers = list(map(float, path.read_text().strip().split()))
        return np.array(list(zip(numbers[::2], numbers[1::2])))
    except:
        return None

def load_dlib_landmarks(path):
    try:
        lines = path.read_text().strip().splitlines()
        return np.array([tuple(map(int, line.strip().split())) for line in lines])
    except:
        return None

# === Comparison + Visualization ===
results = []
distances_all = []
sample_frames = random.sample(list(dlib_landmark_dir.glob("*.txt")), 5)

for dlib_file in sorted(dlib_landmark_dir.glob("*.txt")):
    name = dlib_file.stem
    if "_frame_" not in name:
        continue
    video_id, frame_id = name.split("_frame_")
    s2_file = s2_landmark_base / video_id / f"frame_{frame_id}.txt"
    img_path = image_base / video_id / f"frame_{frame_id}.jpg"

    if not s2_file.exists() or not dlib_file.exists() or not img_path.exists():
        continue

    s2_landmarks = load_s2_landmarks(s2_file)
    dlib_landmarks = load_dlib_landmarks(dlib_file)

    if s2_landmarks is None or dlib_landmarks is None:
        continue

    s2_eye_mouth = s2_landmarks[EYE_MOUTH_IDX]
    dlib_eye_mouth = dlib_landmarks[EYE_MOUTH_IDX]

    if s2_eye_mouth.shape != (32, 2) or dlib_eye_mouth.shape != (32, 2):
        continue

    distances = np.linalg.norm(s2_eye_mouth - dlib_eye_mouth, axis=1)
    avg_dist = np.mean(distances)
    results.append((name, avg_dist))
    distances_all.extend(distances)

    # === Visualize few
    if dlib_file in sample_frames:
        img = cv2.imread(str(img_path))
        if img is not None:
            for (x, y) in s2_eye_mouth:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)  # S2 - blue
            for (x, y) in dlib_eye_mouth:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)  # Dlib - green
            plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Comparison: {name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

# === Histogram of distances ===
plt.figure(figsize=(7, 4))
plt.hist(distances_all, bins=30, color='skyblue', edgecolor='black')
plt.title("Euclidean Distances (Eyes + Mouth Landmarks)")
plt.xlabel("Distance (pixels)")
plt.ylabel("Frequency")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# === Save CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Avg_EyeMouth_Distance"])
    writer.writerows(results)

# === Print Summary ===
print("\n📊 === SUMMARY ===")
print(f"✅ Frames Compared: {len(results)}")
print(f"📉 Mean Distance  : {round(np.mean(distances_all), 2)} px")
print(f"📈 Max Distance   : {round(np.max(distances_all), 2)} px")
print(f"📉 Min Distance   : {round(np.min(distances_all), 2)} px")
print(f"💾 CSV Saved To   : {output_csv}")
