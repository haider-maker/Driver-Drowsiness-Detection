import dlib
import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt

# === CONFIG ===
predictor_path = "shape_predictor_ir.dat"
valid_xml_path = "val.xml"
base_image_dir = Path("data/cropped_faces")
target_video_prefix = "6-2"  # You can change this to "4-3"
num_samples = 5

# === Load Dlib Predictor ===
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# === Parse XML ===
tree = ET.parse(valid_xml_path)
root = tree.getroot()
images = root.findall("images/image")

# === Filter frames from specific video
video_frames = [img.attrib["file"] for img in images if Path(img.attrib["file"]).name.startswith(target_video_prefix)]

print(f"🎯 Found {len(video_frames)} frames from video {target_video_prefix}")
if len(video_frames) == 0:
    exit("🚫 No frames found. Check file naming or prefix.")

# === Pick random samples
random_samples = random.sample(video_frames, min(num_samples, len(video_frames)))

for img_path in random_samples:
    full_img_path = base_image_dir / Path(img_path).name
    img = cv2.imread(str(full_img_path))
    if img is None:
        print(f"❌ Failed to load: {full_img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face manually using bounding box from XML (if available)
    dets = detector(gray, 1)
    if len(dets) == 0:
        print(f"⚠️ No face detected in: {full_img_path.name}")
        continue

    shape = predictor(gray, dets[0])

    # ✅ DRAW ALL 68 LANDMARKS
    for i in range(68):  # not shape.num_parts, be explicit
        part = shape.part(i)
        cv2.circle(img, (part.x, part.y), 1, (0, 255, 0), -1)

    # === Show image
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Landmarks: {Path(img_path).name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
