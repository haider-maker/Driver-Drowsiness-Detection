import dlib
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import matplotlib.pyplot as plt

# === CONFIG ===
predictor_path = "shape_predictor_ir_final.dat"
val_xml_path = "val.xml"
num_samples = 5  # Number of random frames to visualize

# === Load predictor and face detector ===
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# === Parse val.xml ===
tree = ET.parse(val_xml_path)
root = tree.getroot()
images = root.findall("images/image")

# === Randomly sample validation frames ===
sampled_images = random.sample(images, min(num_samples, len(images)))

for image_entry in sampled_images:
    img_path = Path(image_entry.attrib["file"])  # e.g., data/frames_mapped_s2/4-1/frame_1234.jpg

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get bounding box from XML
    boxes = image_entry.findall("box")
    if not boxes:
        print(f"⚠️ No bounding box found in: {img_path.name}")
        continue

    box = boxes[0]
    top, left = int(box.attrib["top"]), int(box.attrib["left"])
    width, height = int(box.attrib["width"]), int(box.attrib["height"])
    rect = dlib.rectangle(left, top, left + width, top + height)

    # Predict landmarks
    shape = predictor(gray, rect)

    # Draw bounding box
    cv2.rectangle(img, (left, top), (left + width, top + height), (255, 0, 0), 2)

    # Draw landmarks
    for i in range(shape.num_parts):
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # Show result
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Landmarks: {img_path.name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
