import os
import math

# === CONFIG ===
landmark_dir = "./landmarks_output"         # Folder containing .txt landmark files
timestamp_dir = "./cropped_faces"           # Folder where .time and .kss files live
output_dir = "./features_output"
os.makedirs(output_dir, exist_ok=True)

# === Constants for Eye and Mouth Landmark Indices ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # [x1, x2, x3, x4, x5, x6]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14]          # Left, Right, TopOuter, TopInner, BottomInner, BottomOuter

# === Helper Functions ===
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def load_landmarks(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

# === DRIVER: Process All Landmark Files ===
processed = 0
for filename in sorted(os.listdir(landmark_dir)):
    if not filename.endswith(".txt"):
        continue

    landmark_path = os.path.join(landmark_dir, filename)
    landmarks = load_landmarks(landmark_path)

    # Validate that all required landmark indices exist
    required_indices = LEFT_EYE + RIGHT_EYE + MOUTH
    if any(idx >= len(landmarks) for idx in required_indices):
        print(f"⚠️ Skipping {filename}, landmarks incomplete.")
        continue

    # Extract eye and mouth points
    left_eye = [landmarks[i] for i in LEFT_EYE]
    right_eye = [landmarks[i] for i in RIGHT_EYE]
    mouth = [landmarks[i] for i in MOUTH]

    # Compute EAR and MAR
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

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0
    mar = compute_mar(mouth)

    # Save EAR and MAR
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        f.write(f"{ear:.6f} {mar:.6f}\n")

    processed += 1

print(f"✅ EAR and MAR extracted for {processed} frames.")
