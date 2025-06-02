import os
import shutil
from pathlib import Path

# Define folders to create
folders = {
    "scripts": [".py"],
    "data": [
        "frames", "frames_test_data", "frames_mapped_s2",
        "cropped_faces", "cropped_faces_test_data",
        "features_output", "features_output_s2", "features_filtered",
        "landmarks_output", "landmarks_output_test_data",
        "window_pre-req", "window_pre-req_s2"
    ],
    "csv": [".csv"],
    "models": [".pkl"],
    "results": [
        "confusion_matrix.png", "rf_confusion_matrix.png",
        "rule_based_confusion_matrix.png",
        "classification_report.txt", "rf_classification_report.txt",
        "yawn_detected_frames.csv", "validation_plots"
    ],
    "yolo": ["yolov8n.pt", "yolo-labels", "yolo_dataset", "data.yaml"]
}

# Move files and folders into their respective directories
def move_to_folder(item, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    try:
        shutil.move(item, os.path.join(target_folder, os.path.basename(item)))
    except Exception as e:
        print(f"❌ Failed to move {item}: {e}")

cwd = Path(".")
all_items = [p for p in cwd.iterdir() if p.name not in [".git", ".gitignore"]]

for item in all_items:
    # Check folders to move
    for folder, patterns in folders.items():
        # Move by folder name match
        if item.is_dir() and item.name in patterns:
            move_to_folder(str(item), folder)
            break
        # Move by file extension or filename
        if item.is_file():
            if item.suffix in patterns or item.name in patterns:
                move_to_folder(str(item), folder)
                break

print("✅ Project structure cleaned successfully.")
