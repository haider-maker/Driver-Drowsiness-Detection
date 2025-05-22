import cv2
import mediapipe as mp
import os

# === MediaPipe Initialization ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # includes iris & lips
    min_detection_confidence=0.5
)

# === CONFIG ===
input_folder = "cropped_faces"          # Folder of IR cropped images
output_folder = "landmarks_output"      # Output folder for landmark .txt files
os.makedirs(output_folder, exist_ok=True)

count = 0
total = 0

# === Loop through all IR images ===
for filename in os.listdir(input_folder):
    if not filename.endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)

    # Load grayscale (IR) image
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"⚠️ Could not read image: {filename}")
        continue

    # Convert grayscale → 3-channel RGB (required by MediaPipe)
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Detect landmarks
    results = face_mesh.process(rgb_image)

    total += 1
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        # Save all 468 landmark points (x, y) as normalized coordinates
        with open(os.path.join(output_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt")), "w") as f:
            for lm in landmarks.landmark:
                f.write(f"{lm.x:.6f} {lm.y:.6f}\n")

        count += 1
        print(f"Processed: {filename}")
    else:
        print(f"No face detected in: {filename}")

print(f" All IR images processed with MediaPipe. Success Rate: {count/total*100}")
