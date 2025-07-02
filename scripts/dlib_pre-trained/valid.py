import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# === CONFIGURATION ===
dlib_landmark_dir = Path("landmarks_dlib_pretrained_simple")  # predicted landmarks
s2_landmark_dir = Path("data/frames_mapped_s2/4-3")        # ground-truth landmarks
frame_dir = Path("data/frames/4-3")                        # IR images

# === FUNCTIONS ===
def load_dlib_landmarks(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            return np.array([tuple(map(int, line.strip().split())) for line in lines])
    except:
        return None

def load_s2_landmarks(path):
    try:
        with open(path, 'r') as f:
            nums = list(map(float, f.read().strip().split()))
            return np.array(list(zip(nums[::2], nums[1::2])))
    except:
        return None

# === 1. RANDOM FRAME VISUALIZATION ===
landmark_files = sorted(dlib_landmark_dir.glob("*.txt"))
sample_files = random.sample(landmark_files, min(5, len(landmark_files)))

for file in sample_files:
    frame_name = file.stem
    img_path = frame_dir / (frame_name + ".jpg")
    s2_path = s2_landmark_dir / ("frame_" + frame_name.split("_")[-1] + ".txt")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    dlib_lm = load_dlib_landmarks(file)
    s2_lm = load_s2_landmarks(s2_path)
    if dlib_lm is None or s2_lm is None:
        continue

    for (x, y) in dlib_lm:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)  # Green = Dlib
    for (x, y) in s2_lm:
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)  # Red = S2

    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Landmark Overlay: {frame_name} (Green: Dlib, Red: S2)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === 2. MEAN EUCLIDEAN DISTANCE (OVERALL + FRAME-WISE) ===
all_distances = []            # all individual landmark distances
frame_distances = []          # mean distance per frame
frame_names = []              # frame names for plotting
num_valid_frames = 0

for file in landmark_files:
    frame_name = file.stem
    s2_path = s2_landmark_dir / ("frame_" + frame_name.split("_")[-1] + ".txt")

    dlib_lm = load_dlib_landmarks(file)
    s2_lm = load_s2_landmarks(s2_path)
    if dlib_lm is None or s2_lm is None or len(dlib_lm) != len(s2_lm):
        continue

    distances = np.linalg.norm(dlib_lm - s2_lm, axis=1)
    all_distances.extend(distances)

    frame_mean = np.mean(distances)
    frame_distances.append(frame_mean)
    frame_names.append(frame_name)
    num_valid_frames += 1

if all_distances:
    mean_distance = np.mean(all_distances)
    print(f"\n✅ Overall Mean Euclidean Distance (all landmark pairs): {mean_distance:.2f} pixels")
    print(f"✅ Based on {num_valid_frames} valid frames and {len(all_distances)} total landmark pairs")

    # Plot per-frame mean distances
    plt.figure(figsize=(10, 4))
    plt.plot(frame_distances, label="Per-frame Mean Distance", color="purple")
    plt.title("Mean Euclidean Distance per Frame (Dlib vs S2)")
    plt.xlabel("Frame Index")
    plt.ylabel("Distance (pixels)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No valid landmark pairs found across frames.")
