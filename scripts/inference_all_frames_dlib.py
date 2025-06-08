import dlib
import cv2
from pathlib import Path

# === CONFIG ===
model_path = "shape_predictor_ir_final.dat"
image_root = Path("data/frames_mapped_s2")
output_dir = Path("data/landmarks_dlib")
output_dir.mkdir(parents=True, exist_ok=True)

predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

# === Inference over all frames ===
total_processed = 0
for subfolder in image_root.iterdir():
    if not subfolder.is_dir():
        continue
    video_id = subfolder.name  # e.g., "1-1"

    for image_file in sorted(subfolder.glob("*.jpg")):
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"❌ Could not load: {image_file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        if len(dets) == 0:
            print(f"⚠️ No face found in {image_file.name}")
            continue

        shape = predictor(gray, dets[0])

        # Save as: 1-1_frame_3567.txt
        output_filename = f"{video_id}_{image_file.stem}.txt"
        output_path = output_dir / output_filename

        with open(output_path, "w") as f:
            for i in range(shape.num_parts):
                x, y = shape.part(i).x, shape.part(i).y
                f.write(f"{x} {y}\n")

        total_processed += 1

print(f"✅ Saved {total_processed} landmark files to {output_dir}")
