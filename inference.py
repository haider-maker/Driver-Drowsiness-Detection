import os
import cv2
from ultralytics import YOLO

# === CONFIG ===
model_path = "C:/Users/Haider Ali Shahid/runs/detect/train10/weights/best.pt"
test_images_dir = "D:/drozy-dataset/yolo_dataset/images/test"
cropped_output_dir = "cropped_faces"

# Create output folder if it doesn't exist
os.makedirs(cropped_output_dir, exist_ok=True)

# === Load the trained model ===
model = YOLO(model_path)

# === Run prediction on all test images ===
results = model.predict(source=test_images_dir, save=True, imgsz=640, verbose=False)

# === Process results: crop and save faces ===
for result in results:
    # Get original image path
    orig_path = result.path
    filename = os.path.basename(orig_path)

    # Load original image
    image = cv2.imread(orig_path)
    h, w = image.shape[:2]

    # Process each detection
    for i, box in enumerate(result.boxes.xywh):  # format: (x_center, y_center, width, height)
        x_center, y_center, bw, bh = box.cpu().numpy()

        # Convert to pixel coordinates
        x_min = int((x_center - bw / 2))
        y_min = int((y_center - bh / 2))
        x_max = int((x_center + bw / 2))
        y_max = int((y_center + bh / 2))

        # Clip coordinates to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Crop and save the face
        cropped_face = image[y_min:y_max, x_min:x_max]
        crop_filename = os.path.join(cropped_output_dir, f"{os.path.splitext(filename)[0]}_face{i}.jpg")
        cv2.imwrite(crop_filename, cropped_face)

print("✅ Inference and cropping completed.")
