import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import shutil

# === CONFIG ===
model_path = "C:/Users/Haider Ali Shahid/runs/detect/train10/weights/best.pt"
frames_dir = Path("./frames")
cropped_output_dir = Path("./cropped_faces")
imgsz = 512

# === Load model
model = YOLO(model_path)

# === Create output directory
cropped_output_dir.mkdir(exist_ok=True)

# === Process each subfolder
for subfolder in sorted(frames_dir.iterdir()):
    if not subfolder.is_dir():
        continue

    print(f"🔍 Processing folder: {subfolder.name}")
    frame_paths = sorted(subfolder.glob("frame_*.jpg"))

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"❌ Could not read {frame_path}")
            continue

        h, w = image.shape[:2]

        # Run YOLO inference on one image at a time
        results = model.predict(source=str(frame_path), imgsz=imgsz, save=False, stream=False, verbose=False)

        # Extract boxes and crop
        for result in results:
            for i, box in enumerate(result.boxes.xywh):
                x_center, y_center, bw, bh = box.cpu().numpy()

                x_min = int(x_center - bw / 2)
                y_min = int(y_center - bh / 2)
                x_max = int(x_center + bw / 2)
                y_max = int(y_center + bh / 2)

                # Clip
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                cropped = image[y_min:y_max, x_min:x_max]
                save_name = f"{subfolder.name}_{frame_path.stem}_face{i}.jpg"
                cv2.imwrite(str(cropped_output_dir / save_name), cropped)

        # Copy .time
        for ext in [".time", ".kss"]:
            src = frame_path.with_suffix(ext)
            dst = cropped_output_dir / f"{subfolder.name}_{frame_path.stem}{ext}"
            if src.exists():
                shutil.copy(src, dst)

print("✅ All folders processed.")
