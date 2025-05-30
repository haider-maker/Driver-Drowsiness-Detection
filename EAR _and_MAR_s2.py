import os
import math

# === CONFIGURATION ===
s2_base_dir = "./frames_mapped_s2"        # Base directory with subfolders
output_dir = "./features_output_s2"       # Flat output directory
os.makedirs(output_dir, exist_ok=True)

# === Landmark indices (CLM/dlib-style 68 landmarks) ===
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 54, 51, 62, 66, 57]  # Left, Right, TopOuter, TopInner, BottomInner, BottomOuter

# === Helper Functions ===
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def parse_landmark_file(filepath):
    with open(filepath, 'r') as f:
        coords = list(map(float, f.read().strip().split()))
    if len(coords) != 136:
        return None
    return [tuple(coords[i:i+2]) for i in range(0, len(coords), 2)]

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth):
    A = euclidean(mouth[2], mouth[4])
    B = euclidean(mouth[3], mouth[5])
    C = euclidean(mouth[0], mouth[1])
    return (A + B) / (2.0 * C)

# === Walk through each subfolder and process .txt annotations ===
processed = 0
skipped = 0

for subfolder in sorted(os.listdir(s2_base_dir)):
    subfolder_path = os.path.join(s2_base_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print(f"\n📂 Processing subfolder: {subfolder}")
    file_count = 0

    for filename in sorted(os.listdir(subfolder_path)):
        if not filename.endswith(".txt"):
            continue

        landmark_path = os.path.join(subfolder_path, filename)
        landmarks = parse_landmark_file(landmark_path)

        if landmarks is None or len(landmarks) != 68:
            print(f"⚠️ Skipped (bad landmarks): {landmark_path}")
            skipped += 1
            continue

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]
        mouth = [landmarks[i] for i in MOUTH]

        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
        mar = compute_mar(mouth)

        out_file = os.path.join(output_dir, f"{subfolder}_{filename}")
        with open(out_file, "w") as f:
            f.write(f"{ear:.6f} {mar:.6f}\n")

        processed += 1
        file_count += 1

        # 🔁 Progress checkpoint every 500 files
        if file_count % 500 == 0:
            print(f"   └── Processed {file_count} files in '{subfolder}'...")

print(f"\n✅ Done. Processed: {processed} | Skipped: {skipped}")
