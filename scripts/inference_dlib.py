import dlib
import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt

# === CONFIG ===
predictor_path = "shape_predictor_ir.dat"
valid_xml_path = "val.xml"
base_image_dir = Path("D:\drozy-dataset\Driver-Drowsiness-Detection\data\cropped_faces")  # Where all images are stored
target_video_prefix = "6-2"
num_samples = 10  # Number of samples to visualize

# === Load Dlib Predictor ===
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# === Parse valid.xml ===
tree = ET.parse(valid_xml_path)
root = tree.getroot()
images = root.findall("images/image")

# === Filter for video 4-3 ===
video_frames = [
    img.attrib["file"] for img in images if Path(img.attrib["file"]).name.startswith(target_video_prefix)
]
print(f"🎯 Found {len(video_frames)} frames from video {target_video_prefix}")

# === Pick random samples ===
random_samples = random.sample(video_frames, min(num_samples, len(video_frames)))

for img_path in random_samples:
    full_img_path = base_image_dir / Path(img_path).name
    img = cv2.imread(str(full_img_path))

    if img is None:
        print(f"❌ Failed to load: {full_img_path}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face (if bounding boxes not available in XML)
    h, w = gray.shape[:2]
    rect = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    shape = predictor(gray, rect)
    # Draw landmarks
    for i in range(0, shape.num_parts):
        x = shape.part(i).x
        y = shape.part(i).y
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # Show using matplotlib
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Landmarks: {Path(img_path).name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
