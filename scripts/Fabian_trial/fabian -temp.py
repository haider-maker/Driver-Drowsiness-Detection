from pathlib import Path
import dlib
import cv2

# === CONFIGURATION ===
model_path = "shape_predictor_68_face_landmarks.dat"

subject = "Fabian"
video_id = "KSS_9_Vid_1"

input_images_dir = Path(f"data/Sync_Extracted_Data/{subject}/{video_id}/sync_images")
output_video_dir = Path(f"data/Subjects_landmarks_dlib_pretrained/{subject}/{video_id}")
output_video_dir.mkdir(parents=True, exist_ok=True)

# === Load dlib tools ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# === CLAHE ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# === Stats ===
total_processed = 0
total_skipped = 0

image_paths = sorted(input_images_dir.glob("*.png"))
print(f"🎞 Found {len(image_paths)} images in {video_id}")

for img_path in image_paths:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Failed to read: {img_path.name}")
        total_skipped += 1
        continue

    # Handle grayscale vs color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Smooth noise
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Enhance contrast
    gray = clahe.apply(gray)

    # Increase upsample for small faces
    dets = detector(gray, 2)

    if len(dets) == 0:
        print(f"⚠️ No face detected in: {img_path.name}")
        total_skipped += 1
        continue

    face_found = False
    for det in dets:
        shape = predictor(gray, det)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        out_file = output_video_dir / f"{img_path.stem}.txt"
        with open(out_file, "w") as f:
            for (x, y) in landmarks:
                f.write(f"{x} {y}\n")

        face_found = True
        break  # comment this if you want ALL faces saved

    if face_found:
        total_processed += 1
    else:
        total_skipped += 1

    if total_processed % 50 == 0 and total_processed > 0:
        print(f"      ✅ Processed {total_processed} images so far...")

print(f"\n🎯 Done. Total Processed: {total_processed}, Skipped: {total_skipped}.")
print(f"✅ Landmarks saved under: {output_video_dir.resolve()}")
