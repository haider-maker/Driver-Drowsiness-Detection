import os
import shutil
from pathlib import Path

# === CONFIGURATION ===
features_dir = Path("data/EAR_MAR_dlib_output")  # Contains EAR + MAR .txt files (with subfolders)
frames_dir = Path("data/frames")                 # Contains KSS and time files (with subfolders)
output_dir = Path("data/window_pre-req_s2")      # Output will mirror features_dir structure
output_dir.mkdir(parents=True, exist_ok=True)

print("📦 Starting feature aggregation...")

# === Stats ===
processed = 0
skipped = 0
total = 0
missing_kss = 0
missing_time = 0
bad_filename = 0

# Recursively process all .txt files
feature_files = sorted(features_dir.rglob("*.txt"))
print(f"🔍 Found {len(feature_files)} files in {features_dir}\n")

for idx, feat_path in enumerate(feature_files, 1):
    total += 1
    file = feat_path.name
    video_id = feat_path.parent.name  # e.g., '1-1'
    frame_name = file.replace(".txt", "")  # e.g., 'frame_2587'

    # Construct full paths
    kss_path = frames_dir / video_id / f"{frame_name}.kss"
    time_path = frames_dir / video_id / f"{frame_name}.time"

    missing_files = []
    if not kss_path.exists():
        missing_kss += 1
        missing_files.append("KSS")
    if not time_path.exists():
        missing_time += 1
        missing_files.append("Time")
    if not feat_path.exists():
        missing_files.append("Feature")

    if missing_files:
        print(f"⚠️ [{idx}] Missing {', '.join(missing_files)} for {file}")
        skipped += 1
        continue

    # Create output subdirectory
    output_subdir = output_dir / video_id
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Define full output paths
    shutil.copy(feat_path, output_subdir / f"{frame_name}.txt")
    shutil.copy(kss_path, output_subdir / f"{frame_name}.kss")
    shutil.copy(time_path, output_subdir / f"{frame_name}.time")

    processed += 1
    if processed % 100 == 0:
        print(f"✅ Processed: {processed} so far...")

# === Final Summary ===
print("\n🎯 Final Report")
print(f"📁 Total feature files found: {len(feature_files)}")
print(f"✅ Combined and saved: {processed}")
print(f"⚠️ Skipped (incomplete): {skipped}")
print(f"   └── Missing KSS files: {missing_kss}")
print(f"   └── Missing Time files: {missing_time}")
print(f"   └── Bad filenames: {bad_filename}")
