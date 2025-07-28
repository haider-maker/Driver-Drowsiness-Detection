from pathlib import Path
import dlib
import cv2

# === CONFIGURATION ===
model_path = "shape_predictor_68_face_landmarks.dat"
input_root = Path("data/Sync_Extracted_Data/Fabian/KSS_9_Vid_1")
output_base = Path("data/Subjects_landmarks_dlib_pretrained")
output_base.mkdir(parents=True, exist_ok=True)

# === Load dlib tools ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# === CLAHE ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# === Stats ===
total_processed = 0
total_skipped = 0

for subject_folder in sorted(input_root.glob("*")):
    if not subject_folder.is_dir():
        continue

    subject = subject_folder.name
    print(f"\n👤 Subject: {subject}")

    for video_folder in sorted(subject_folder.glob("*")):
        if not video_folder.is_dir():
            continue

        video_id = video_folder.name
        images_dir = video_folder / "sync_images"
        if not images_dir.exists():
            print(f"⚠️ No sync_images folder in {video_folder}")
            continue

        output_video_dir = output_base / subject / video_id
        output_video_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(images_dir.glob("*.png"))
        print(f"   🎞 Found {len(image_paths)} images in {video_id}")

        processed = 0
        skipped = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"❌ Failed to read: {img_path.name}")
                skipped += 1
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
                skipped += 1
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
                processed += 1
            else:
                skipped += 1

            if processed % 50 == 0 and processed > 0:
                print(f"      ✅ Processed {processed} images so far...")

        print(f"   📁 {video_id} — Processed: {processed}, Skipped: {skipped}")
        total_processed += processed
        total_skipped += skipped

print(f"\n🎯 All Done. Total Processed: {total_processed}, Skipped: {total_skipped}")
print(f"✅ Landmarks saved under: {output_base.resolve()}")
