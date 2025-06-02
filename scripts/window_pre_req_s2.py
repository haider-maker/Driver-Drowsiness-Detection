import os
import shutil

# === CONFIGURATION ===
features_dir = "./features_output_s2"    # Contains EAR + MAR .txt files
frames_dir = "./frames"                  # Contains KSS and time files
output_dir = "./window_pre-req_s2"       # New flat output folder
os.makedirs(output_dir, exist_ok=True)

print("📦 Starting feature aggregation...")

# === Process all feature files ===
processed = 0
skipped = 0
total = 0
missing_kss = 0
missing_time = 0
bad_filename = 0

feature_files = sorted(os.listdir(features_dir))
print(f"🔍 Found {len(feature_files)} files in {features_dir}\n")

for idx, file in enumerate(feature_files, 1):
    if not file.endswith(".txt"):
        continue

    total += 1

    # Parse video ID and frame name
    try:
        video_id, frame_part = file.split("_frame_")
        frame_name = f"frame_{frame_part.replace('.txt', '')}"
    except Exception as e:
        print(f"❌ [{idx}] Failed to parse filename: {file}")
        bad_filename += 1
        skipped += 1
        continue

    # Paths to source files
    kss_path = os.path.join(frames_dir, video_id, f"{frame_name}.kss")
    time_path = os.path.join(frames_dir, video_id, f"{frame_name}.time")
    feat_path = os.path.join(features_dir, file)

    missing_files = []
    if not os.path.exists(kss_path):
        missing_kss += 1
        missing_files.append("KSS")
    if not os.path.exists(time_path):
        missing_time += 1
        missing_files.append("Time")
    if not os.path.exists(feat_path):
        missing_files.append("Feature")

    if missing_files:
        print(f"⚠️ [{idx}] Missing {', '.join(missing_files)} for {file}")
        skipped += 1
        continue

    # Output filenames
    base_name = f"{video_id}_{frame_name}"
    shutil.copy(feat_path, os.path.join(output_dir, f"{base_name}.txt"))
    shutil.copy(kss_path, os.path.join(output_dir, f"{base_name}.kss"))
    shutil.copy(time_path, os.path.join(output_dir, f"{base_name}.time"))

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
