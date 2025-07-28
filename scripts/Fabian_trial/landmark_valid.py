import random
import cv2
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
subject = "Theresa"
video_id = "KSS_8_Vid_1"

# Landmarks folder
landmarks_path = Path(f"data/Subjects_landmarks_dlib_pretrained/{subject}/{video_id}")

# Images folder
frames_path = Path(f"data/Sync_Extracted_Data/{subject}/{video_id}/sync_images")

# Where to save visualizations
visual_output_dir = Path(f"data/{subject}_trail_landmarks_visualized/{video_id}")
visual_output_dir.mkdir(parents=True, exist_ok=True)

# === Load landmark files ===
landmark_files = sorted(landmarks_path.glob("*.txt"))

if not landmark_files:
    print(f"❌ No landmark files found in {landmarks_path}")
    exit()

sample_files = random.sample(landmark_files, min(20, len(landmark_files)))

for lm_file in sample_files:
    # Read landmarks
    landmarks = []
    with open(lm_file, "r") as f:
        for line in f:
            x, y = map(int, line.strip().split())
            landmarks.append((x, y))

    # Load corresponding frame
    frame_name = lm_file.stem + ".png"
    img_path = frames_path / frame_name
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Save to file
    out_path = visual_output_dir / frame_name
    cv2.imwrite(str(out_path), img)
    print(f"Saved visualization: {out_path}")
