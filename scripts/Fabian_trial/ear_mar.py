import os
import math
from pathlib import Path

# === CONFIGURATION ===
landmark_dir = Path("data/Fabian_trail_landmarks_dlib_pretrained")         # Input landmark folders
output_dir = Path("data/Fabian_trail_EAR_MAR_output")                 # Output folder (root)
output_dir.mkdir(parents=True, exist_ok=True)

# === Facial Landmark Indices (68-point format) ===
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 54, 51, 62, 66, 57]  # [left, right, top_outer, top_inner, bottom_inner, bottom_outer]

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def load_landmarks(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    if len(lines) != 68:
        return None
    return [tuple(map(int, line.strip().split())) for line in lines]

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth):
    A = euclidean(mouth[2], mouth[4])  # top_outer to bottom_outer
    B = euclidean(mouth[3], mouth[5])  # top_inner to bottom_inner
    C = euclidean(mouth[0], mouth[1])  # left to right
    return (A + B) / (2.0 * C)

# === Process All Landmark Files Recursively ===
processed = 0
for txt_file in sorted(landmark_dir.rglob("*.txt")):
    landmarks = load_landmarks(txt_file)
    if landmarks is None:
        print(f"⚠️ Skipping {txt_file}: invalid number of landmarks.")
        continue

    # EAR and MAR calculation
    left_eye = [landmarks[i] for i in LEFT_EYE]
    right_eye = [landmarks[i] for i in RIGHT_EYE]
    mouth = [landmarks[i] for i in MOUTH]

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0
    mar = compute_mar(mouth)

    # Preserve subfolder structure
    relative_path = txt_file.relative_to(landmark_dir)
    out_path = output_dir / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"{ear:.6f} {mar:.6f}\n")

    processed += 1

print(f"✅ EAR and MAR extracted and saved for {processed} frames.")
