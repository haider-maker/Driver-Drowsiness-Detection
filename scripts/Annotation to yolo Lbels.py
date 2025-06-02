import os

input_base_dir = "./frames"  
output_base_dir = "./yolo-labels"             
image_width = 512
image_height = 424
class_id = 0

def convert_folder_landmarks_to_yolo(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue
        
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r') as f:
            line = f.read().strip()
            coords = list(map(float, line.split()))
        
        if len(coords) != 136:
            print(f" Skipping {input_path}: does not contain 68 (x, y) points.")
            continue
        
        xs = coords[::2]
        ys = coords[1::2]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as out_f:
            out_f.write(yolo_line)

    print(f" Processed folder: {input_dir}")

# === WALK THROUGH ALL SUBFOLDERS ===
for root, dirs, files in os.walk(input_base_dir):
    # Only process directories with .txt files (skip the base directory itself)
    if root == input_base_dir:
        continue

    # Map input folder to output folder
    relative_path = os.path.relpath(root, input_base_dir)
    output_dir = os.path.join(output_base_dir, relative_path)

    convert_folder_landmarks_to_yolo(root, output_dir)

print("All folders processed successfully!")
