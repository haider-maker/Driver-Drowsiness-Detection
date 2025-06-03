import xml.etree.ElementTree as ET

# Path to your train.xml file
xml_path = "train.xml"

# Load and parse the XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Count number of images and landmarks per image
num_images = 0
landmark_counts = []

for image in root.findall("images/image"):
    num_images += 1
    boxes = image.findall("box")
    for box in boxes:
        parts = box.findall("part")
        landmark_counts.append(len(parts))

# Summary
print(f"✅ Total annotated frames: {num_images}")
if landmark_counts:
    print(f"📌 Min landmarks per image: {min(landmark_counts)}")
    print(f"📌 Max landmarks per image: {max(landmark_counts)}")
    print(f"📌 Average landmarks per image: {sum(landmark_counts)/len(landmark_counts):.2f}")
else:
    print("⚠️ No landmarks found in any image.")
