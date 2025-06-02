import cv2
import mediapipe as mp
import os
from tqdm import tqdm

# === MediaPipe Initialization ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# === CONFIG ===
input_folder = "cropped_faces"              # Folder with cropped IR images
output_folder = "landmarks_output"          # Where .txt files of landmarks will go
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

count = 0

for filename in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, filename)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"❌ Skipping unreadable image: {filename}")
        continue

    # Convert to 3-channel RGB (needed by MediaPipe)
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        output_path = os.path.join(output_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(output_path, "w") as f:
            for lm in landmarks.landmark:
                f.write(f"{lm.x:.6f} {lm.y:.6f}\n")

        count += 1
    else:
        print(f"⚠️ No face detected in: {filename}")

print(f"\n✅ Completed. {count}/{len(image_files)} images had detectable landmarks.")
