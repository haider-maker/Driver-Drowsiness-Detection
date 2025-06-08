import os
import xml.etree.ElementTree as ET
from pathlib import Path

# === CONFIG ===
base_dir = Path("data/frames_mapped_s2")
output_xml = Path("full_frame_dlib_dataset.xml")

print(f"📂 Checking base directory: {base_dir}")
if not base_dir.exists() or not base_dir.is_dir():
    print(f"❌ ERROR: Base directory not found: {base_dir}")
    exit(1)

# === Initialize XML Root ===
dataset = ET.Element("dataset")
ET.SubElement(dataset, "name").text = "Dlib Full Frame Dataset"
ET.SubElement(dataset, "comment").text = "Generated from full-frame S2 annotations"
images_element = ET.SubElement(dataset, "images")

subfolder_count = 0
file_count = 0
skipped_missing_img = 0
skipped_invalid_landmarks = 0
added_images = 0

# === Iterate Over Subfolders ===
for subfolder in sorted(base_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    subfolder_count += 1
    print(f"\n📁 Processing subfolder: {subfolder}")

    for file in sorted(subfolder.glob("*.txt")):
        file_count += 1
        txt_path = file
        img_path = txt_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"⚠️ Skipping: Image not found for {txt_path}")
            skipped_missing_img += 1
            continue

        # Load S2 landmarks from .txt file
        try:
            with txt_path.open("r") as f:
                coords = list(map(float, f.read().strip().split()))
        except Exception as e:
            print(f"❌ Error reading {txt_path}: {e}")
            continue

        if len(coords) != 68 * 2:
            print(f"⚠️ Skipping: Invalid landmark count in {txt_path} → {len(coords)} values")
            skipped_invalid_landmarks += 1
            continue  # Skip invalid annotation

        # Compute bounding box around landmarks
        xs = coords[::2]
        ys = coords[1::2]
        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs))
        bottom = int(max(ys))
        width = right - left
        height = bottom - top

        # Create <image> entry
        rel_img_path = str(img_path).replace("\\", "/")  # Ensure compatibility
        image_elem = ET.SubElement(images_element, "image", file=rel_img_path)
        box_elem = ET.SubElement(image_elem, "box", top=str(top), left=str(left),
                                 width=str(width), height=str(height))

        # Add all 68 landmarks
        for i in range(68):
            x = int(xs[i])
            y = int(ys[i])
            ET.SubElement(box_elem, "part", name=str(i), x=str(x), y=str(y))

        added_images += 1
        print(f"✅ Added image: {rel_img_path}")

# === Write XML ===
try:
    tree = ET.ElementTree(dataset)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"\n💾 XML written to: {output_xml.resolve()}")
except Exception as e:
    print(f"❌ Failed to write XML: {e}")
    exit(1)

# === Summary ===
print("\n📊 Summary:")
print(f"  Subfolders scanned: {subfolder_count}")
print(f"  TXT files found: {file_count}")
print(f"  Images added: {added_images}")
print(f"  Skipped (missing image): {skipped_missing_img}")
print(f"  Skipped (invalid landmarks): {skipped_invalid_landmarks}")
