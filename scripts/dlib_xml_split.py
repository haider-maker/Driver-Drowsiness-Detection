from pathlib import Path
import xml.etree.ElementTree as ET

# === CONFIG ===
input_xml_path = Path("training_with_face_landmarks.xml")
train_output_path = Path("train.xml")
val_output_path = Path("val.xml")
train_ratio = 0.8  # 80% training, 20% validation

# === Parse the input XML ===
tree = ET.parse(input_xml_path)
root = tree.getroot()

# Find all <image> elements
images = root.find("images")
image_elements = images.findall("image")

print(f"🔍 Total images found in XML: {len(image_elements)}")

# Sanity check: If no images, something went wrong
if not image_elements:
    raise ValueError("❌ No <image> entries found in the input XML file!")

# === Split the dataset ===
split_index = int(len(image_elements) * train_ratio)
train_images = image_elements[:split_index]
val_images = image_elements[split_index:]

print(f"✅ Training samples: {len(train_images)}")
print(f"✅ Validation samples: {len(val_images)}")

# === Helper: Create a new XML tree ===
def create_dataset_xml(images_subset):
    dataset = ET.Element("dataset")
    ET.SubElement(dataset, "name").text = "Split Dataset"
    ET.SubElement(dataset, "comment").text = "Train/Val Split"
    images_tag = ET.SubElement(dataset, "images")
    for img in images_subset:
        images_tag.append(img)
    return ET.ElementTree(dataset)

# === Save to XML files ===
create_dataset_xml(train_images).write(train_output_path, encoding="utf-8", xml_declaration=True)
create_dataset_xml(val_images).write(val_output_path, encoding="utf-8", xml_declaration=True)

print(f"💾 Files saved: {train_output_path} and {val_output_path}")
