import os
import math
import cv2

# === CONFIG ===
landmark_dir = "./landmarks_output"         # Folder with MediaPipe landmark .txt files (normalized)
image_dir = "./cropped_faces"               # Folder with corresponding images
output_dir = "./features_test_output"            # Output EAR/MAR per frame
os.makedirs(output_dir, exist_ok=True)

# === Landmark Indices ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14]  # Left, Right, TopOuter, TopInner, BottomInner, BottomOuter

# === Helper functions ===
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def load_landmarks(landmark_path, image_shape):
    h, w = image_shape[:2]
    landmarks = []
    with open(landmark_path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split())
            landmarks.append([x * w, y * h])  # Convert to pixel coords
    return landmarks

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

# === Main processing ===
processed = 0
skipped = 0

for filename in sorted(os.listdir(landmark_dir)):
    if not filename.endswith(".txt"):
        continue

    base_name = filename.replace(".txt", "")
    landmark_path = os.path.join(landmark_dir, filename)
    image_path = os.path.join(image_dir, base_name + ".jpg")

    if not os.path.exists(image_path):
        print(f"❌ Image not found for {base_name}")
        skipped += 1
        continue

    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        skipped += 1
        continue

    # Load and convert landmarks to pixel coordinates
    landmarks = load_landmarks(landmark_path, image.shape)
    if len(landmarks) < 468:
        print(f"⚠️ Incomplete landmarks in {filename}, skipping.")
        skipped += 1
        continue

    # Extract points and compute features
    left_eye = [landmarks[i] for i in LEFT_EYE]
    right_eye = [landmarks[i] for i in RIGHT_EYE]
    mouth = [landmarks[i] for i in MOUTH]

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0
    mar = compute_mar(mouth)

    # Save EAR and MAR
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        f.write(f"{ear:.4f} {mar:.4f}\n")

    processed += 1

print(f"\n✅ EAR and MAR extracted for {processed} frames.")
if skipped > 0:
    print(f"⚠️ Skipped {skipped} frames due to errors or missing files.")
