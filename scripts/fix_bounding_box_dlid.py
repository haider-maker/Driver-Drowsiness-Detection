import xml.etree.ElementTree as ET
import os

# === CONFIG ===
input_xml = "training_with_face_landmarks.xml"
output_xml = "training_with_face_landmarks_boxed.xml"

print("🔍 Checking if input XML exists...")
if not os.path.exists(input_xml):
    print(f"❌ Input XML file not found: {input_xml}")
    exit(1)

# === Parse original XML ===
try:
    print(f"📂 Parsing XML file: {input_xml}")
    tree = ET.parse(input_xml)
    root = tree.getroot()
except ET.ParseError as e:
    print(f"❌ XML parsing failed: {e}")
    exit(1)

corrected = 0
total_images = 0
skipped_no_box = 0
skipped_landmarks = 0

for img in root.findall(".//image"):
    total_images += 1
    if "file" not in img.attrib:
        print(f"⚠️ Image tag missing 'file' attribute.")
        continue

    box = img.find("box")
    if box is None:
        print(f"⚠️ Skipping {img.attrib['file']}, no bounding box found.")
        skipped_no_box += 1
        continue

    parts = box.findall("part")
    if len(parts) != 68:
        print(f"⚠️ Skipping {img.attrib['file']}, only {len(parts)} landmarks found.")
        skipped_landmarks += 1
        continue

    try:
        xs = [int(p.attrib['x']) for p in parts]
        ys = [int(p.attrib['y']) for p in parts]
    except Exception as e:
        print(f"❌ Error parsing coordinates in {img.attrib['file']}: {e}")
        continue

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    width = max_x - min_x
    height = max_y - min_y

    # Update the box attributes
    box.set("left", str(min_x))
    box.set("top", str(min_y))
    box.set("width", str(width))
    box.set("height", str(height))
    corrected += 1

# === Save output ===
try:
    tree.write(output_xml)
    print(f"✅ Fixed bounding boxes for {corrected} images.")
    print(f"💾 Saved corrected XML to: {output_xml}")
except Exception as e:
    print(f"❌ Failed to write output XML: {e}")

# === Summary ===
print(f"\n📊 Summary:")
print(f"  Total images processed: {total_images}")
print(f"  Corrected: {corrected}")
print(f"  Skipped (no box): {skipped_no_box}")
print(f"  Skipped (wrong landmarks): {skipped_landmarks}")
