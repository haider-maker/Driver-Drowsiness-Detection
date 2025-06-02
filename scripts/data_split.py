import os
import shutil
import random

images_base_dir = "./frames"            
labels_base_dir = "./yolo-labels"       
output_base_dir = "./yolo_dataset"      

split_ratio = (0.7, 0.15, 0.15)  

# === Create output folder structure ===
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'labels', split), exist_ok=True)

# === Collect all image-label pairs ===
image_label_pairs = []

for subject in os.listdir(images_base_dir):
    image_folder = os.path.join(images_base_dir, subject)
    label_folder = os.path.join(labels_base_dir, subject)

    if not os.path.isdir(image_folder) or not os.path.isdir(label_folder):
        continue

    for file in os.listdir(image_folder):
        if not file.endswith(".jpg"):
            continue

        image_path = os.path.join(image_folder, file)
        label_path = os.path.join(label_folder, file.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            image_label_pairs.append((image_path, label_path))

# === Shuffle and split ===
random.shuffle(image_label_pairs)
total = len(image_label_pairs)

train_end = int(split_ratio[0] * total)
val_end = train_end + int(split_ratio[1] * total)

splits = {
    'train': image_label_pairs[:train_end],
    'val': image_label_pairs[train_end:val_end],
    'test': image_label_pairs[val_end:]
}

# === Copy files into split folders ===
for split, pairs in splits.items():
    for img_path, label_path in pairs:
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(label_path)

        shutil.copy(img_path, os.path.join(output_base_dir, 'images', split, img_name))
        shutil.copy(label_path, os.path.join(output_base_dir, 'labels', split, lbl_name))

    print(f" {split.upper()}: {len(pairs)} samples")

print("Dataset split complete and YOLOv8-ready!")
