import os
from pathlib import Path
import xml.etree.ElementTree as ET

# === CONFIG ===
image_folder = Path("data/cropped_faces")
output_xml = "training_with_face_landmarks.xml"
num_landmarks = 68

# === Collect all (image, landmark) pairs ===
image_files = sorted(image_folder.glob("*_face0.jpg"))
pairs = []

for img_file in image_files:
    base = img_file.stem  # e.g., 1-1_frame_3066_face0
    landmark_file = image_folder / f"{base}.txt"

    if landmark_file.exists():
        with open(landmark_file, "r") as f:
            coords = f.read().strip().split()
            if len(coords) != num_landmarks * 2:
                print(f"⚠️ Skipping {base}: Found {len(coords)//2} points instead of {num_landmarks}")
                continue
            coords = list(map(float, coords))
            points = [(int(coords[i]), int(coords[i + 1])) for i in range(0, len(coords), 2)]
            pairs.append((img_file, points))
    else:
        print(f"❌ Landmark file missing for {base}")

print(f"✅ Found {len(pairs)} valid image/landmark pairs.")

# === Build XML ===
dataset = ET.Element("dataset")
ET.SubElement(dataset, "name").text = "Driver Drowsiness Dataset"
ET.SubElement(dataset, "comment").text = "Converted from S2 annotations"
images_tag = ET.SubElement(dataset, "images")

for img_path, landmarks in pairs:
    image_tag = ET.SubElement(images_tag, "image", file=str(img_path))

    box = ET.SubElement(image_tag, "box", top="0", left="0", width="1", height="1")
    for i, (x, y) in enumerate(landmarks):
        ET.SubElement(box, "part", name=str(i), x=str(x), y=str(y))

# === Write to file ===
tree = ET.ElementTree(dataset)
tree.write(output_xml, encoding="utf-8", xml_declaration=True)

print(f"\n📦 Dlib XML saved to: {output_xml}")
