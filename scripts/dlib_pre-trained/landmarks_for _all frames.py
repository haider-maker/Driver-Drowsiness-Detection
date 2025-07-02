from pathlib import Path
import dlib
import cv2

# === CONFIGURATION ===
model_path = "shape_predictor_68_face_landmarks.dat"  # Pretrained dlib model
input_root = Path("data/frames")                      # Root folder containing multiple video folders
output_root = Path("data/landmarks_dlib_pretrained")  # Output base folder
output_root.mkdir(parents=True, exist_ok=True)

# === Load dlib tools ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# === Stats ===
total_processed = 0
total_skipped = 0

# === Iterate over video folders ===
for video_folder in sorted(input_root.glob("*")):
    if not video_folder.is_dir():
        continue

    output_video_dir = output_root / video_folder.name
    output_video_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for img_path in sorted(video_folder.glob("frame_*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to read: {img_path.name}")
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

        if len(dets) == 0:
            print(f"⚠️ No face in: {img_path.name}")
            skipped += 1
            continue

        shape = predictor(gray, dets[0])
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Save landmarks to file
        out_file = output_video_dir / f"{img_path.stem}.txt"
        with open(out_file, "w") as f:
            for (x, y) in landmarks:
                f.write(f"{x} {y}\n")

        processed += 1
        if processed % 50 == 0:
            print(f"✅ {video_folder.name}: Processed {processed} images...")

    total_processed += processed
    total_skipped += skipped
    print(f"📁 {video_folder.name} — Processed: {processed}, Skipped: {skipped}")

print(f"\n🎯 Done. Total Processed: {total_processed}, Skipped: {total_skipped}, Output saved in {output_root.resolve()}")
