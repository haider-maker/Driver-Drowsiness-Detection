from pathlib import Path
import dlib
import cv2

# === CONFIGURATION ===
model_path = "shape_predictor_68_face_landmarks.dat"
input_root = Path("data/Fabian_KSS_8_Vid_1")
output_root = Path("data/Fabian_trail_landmarks_dlib_pretrained")
output_root.mkdir(parents=True, exist_ok=True)

# === Load dlib tools ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# === CLAHE ===
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# === Stats ===
total_processed = 0
total_skipped = 0

for img_path in sorted(input_root.glob("frame_*.jpg")):
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

    # OPTIONAL ROI crop (uncomment to try)
    # h, w = gray.shape
    # gray = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

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

        out_file = output_root / f"{img_path.stem}.txt"
        with open(out_file, "w") as f:
            for (x, y) in landmarks:
                f.write(f"{x} {y}\n")

        face_found = True
        break  # comment this if you want ALL faces saved

    if face_found:
        total_processed += 1
    else:
        total_skipped += 1

    if total_processed % 50 == 0:
        print(f"✅ Processed {total_processed} images so far...")

print(f"\n🎯 Done. Total Processed: {total_processed}, Skipped: {total_skipped}. Output saved in {output_root.resolve()}")
