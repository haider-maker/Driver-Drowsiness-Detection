import random
import cv2
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
frames_path = Path("data/Fabian_KSS_8_Vid_1")
landmarks_path = Path("data/Fabian_trail_landmarks_dlib_pretrained")
visual_output_dir = Path("data/Fabian_trail_landmarks_visualized")
visual_output_dir.mkdir(parents=True, exist_ok=True)

# List all landmark files
landmark_files = sorted(landmarks_path.glob("*.txt"))
sample_files = random.sample(landmark_files, min(20, len(landmark_files)))

for lm_file in sample_files:
    # Read landmarks
    landmarks = []
    with open(lm_file, "r") as f:
        for line in f:
            x, y = map(int, line.strip().split())
            landmarks.append((x, y))

    # Load corresponding frame
    frame_name = lm_file.stem + ".jpg"
    img_path = frames_path / frame_name
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Image not found: {img_path}")
        continue

    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Show the image (uncomment if you want a pop-up window)
    # cv2.imshow("Landmarks", img)
    # cv2.waitKey(0)

    # Save to file
    out_path = visual_output_dir / frame_name
    cv2.imwrite(str(out_path), img)
    print(f"Saved visualization: {out_path}")

# cv2.destroyAllWindows()
