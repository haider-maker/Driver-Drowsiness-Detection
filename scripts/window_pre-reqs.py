import shutil
from pathlib import Path

# === CONFIGURATION ===
features_dir = Path("./features_output")
cropped_faces_dir = Path("./cropped_faces")
output_dir = Path("./window_pre-req")

# Create the output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# === MAPPING PROCESS ===
mapped = 0
total = 0

for txt_file in sorted(features_dir.glob("*.txt")):
    total += 1
    base_name = txt_file.stem.split("_face")[0]  # e.g. "9-3_frame_3577"

    # Corresponding .kss and .time filenames
    kss_file = cropped_faces_dir / f"{base_name}.kss"
    time_file = cropped_faces_dir / f"{base_name}.time"

    # Check if both files exist before copying
    if kss_file.exists() and time_file.exists():
        shutil.copy(txt_file, output_dir / txt_file.name)
        shutil.copy(kss_file, output_dir / kss_file.name)
        shutil.copy(time_file, output_dir / time_file.name)
        print(f"✅ Mapped: {base_name}")
        mapped += 1
    else:
        print(f"⚠️ Skipped: {base_name} (Missing .kss or .time)")

print(f"\n🎉 Done! Mapped {mapped}/{total} .txt files into '{output_dir.name}' with corresponding .kss and .time files.")
