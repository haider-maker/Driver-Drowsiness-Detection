import cv2
import os
import random

# === CONFIG ===
images_folder = "cropped_faces"
landmarks_folder = "landmarks_output"
output_folder = "landmark_vis"
os.makedirs(output_folder, exist_ok=True)

# === PARAMETERS ===
num_samples = 10  # How many images to visualize

# === Get list of valid sample filenames ===
all_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
selected_files = random.sample(all_files, min(num_samples, len(all_files)))

for filename in selected_files:
    image_path = os.path.join(images_folder, filename)
    landmark_path = os.path.join(landmarks_folder, filename.replace(".jpg", ".txt"))

    if not os.path.exists(landmark_path):
        print(f" Landmark file missing for: {filename}")
        continue

    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Load landmarks
    with open(landmark_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        x_norm, y_norm = map(float, line.strip().split())
        x = int(x_norm * w)
        y = int(y_norm * h)
        cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Save the visualized image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)
    print(f" Saved: {output_path}")

print(f"\n All {len(selected_files)} images visualized and saved to `{output_folder}`.")
