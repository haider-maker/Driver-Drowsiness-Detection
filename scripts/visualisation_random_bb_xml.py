import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
import random

# === CONFIG ===
xml_path = Path("full_frame_dlib_dataset.xml")
base_dir = Path("data/frames_mapped_s2")
num_samples = 5

# === Load XML and extract image elements ===
tree = ET.parse(xml_path)
root = tree.getroot()
images = root.findall("images/image")

# === Randomly sample image entries ===
sampled_images = random.sample(images, min(num_samples, len(images)))

for idx, img_elem in enumerate(sampled_images):
    img_path = Path(img_elem.attrib["file"])
    full_img_path = Path(img_elem.attrib["file"])

    if not full_img_path.exists():
        print(f"❌ Image not found: {full_img_path}")
        continue

    img = cv2.imread(str(full_img_path))
    if img is None:
        print(f"❌ Failed to read image: {full_img_path}")
        continue

    # Draw bounding box
    box = img_elem.find("box")
    left = int(box.attrib["left"])
    top = int(box.attrib["top"])
    width = int(box.attrib["width"])
    height = int(box.attrib["height"])
    right = left + width
    bottom = top + height
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw landmarks
    for part in box.findall("part"):
        x = int(part.attrib["x"])
        y = int(part.attrib["y"])
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    # ✅ Save the image
    cv2.imwrite(f"annotated_sample_{idx+1}.jpg", img)

    # Display
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"File: {img_path.name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
