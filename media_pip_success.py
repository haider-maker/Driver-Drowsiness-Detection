import os

# === CONFIG ===
images_dir = "cropped_faces"
landmarks_dir = "landmarks_output"

# === Count images and landmark files ===
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
landmark_files = [f for f in os.listdir(landmarks_dir) if f.endswith(".txt")]

total_images = len(image_files)
successful_landmarks = len(landmark_files)

# === Compute success rate ===
if total_images == 0:
    print("❌ No images found.")
else:
    success_rate = (successful_landmarks / total_images) * 100
    print(f"✅ MediaPipe Success Rate: {successful_landmarks}/{total_images} ({success_rate:.2f}%)")
